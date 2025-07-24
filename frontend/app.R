# Enhanced trivago Hotel Offer Ranking Simulator
# Load necessary libraries
library(shiny)
library(shinydashboard)
library(shinyjs)
library(DT)
library(plotly)
library(ggplot2)
library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)
library(readr)

# --- Configuration ---
API_URL <- Sys.getenv("API_URL")
if (is.null(API_URL) || API_URL == "") {
  API_URL <- "http://backend:8001"
  print("API_URL not found in environment, using default: http://backend:8001")
} else {
  print(paste("API_URL from environment:", API_URL))
}

# Additional debugging
print(paste("Final API_URL being used:", API_URL))
print(paste("All environment variables:", paste(names(Sys.getenv()), collapse=", ")))

# Test startup message
print("=== SHINY APP STARTING ===")
print("Libraries loaded successfully")
print("Configuration completed")

# Write to debug log
write(paste("App starting at", Sys.time(), "API_URL:", API_URL), "/tmp/shiny_debug.log", append = TRUE)

# --- Performance Optimization: Data Cache ---
# Global cache for data to avoid repeated API calls
data_cache <- reactiveValues(
  bandit_data = NULL,
  dps_data = NULL,
  conversion_data = NULL,
  market_data = NULL,
  user_data = NULL,
  ranking_data = NULL,
  shapley_data = NULL,
  policy_heatmap = NULL,
  last_bandit_update = NULL,
  last_dps_update = NULL,
  last_conversion_update = NULL,
  last_market_update = NULL,
  last_user_update = NULL,
  last_ranking_update = NULL,
  last_shapley_update = NULL,
  last_heatmap_update = NULL,
  cache_timeout = 300  # 5 minutes cache timeout
)

# --- Helper Functions ---
check_api_connection <- function() {
  tryCatch({
    res <- GET(paste0(API_URL, "/"))
    return(http_status(res)$category == "Success")
  }, error = function(e) {
    return(FALSE)
  })
}

format_currency <- function(x) {
  paste0("$", format(round(x, 2), nsmall = 2))
}

# Cached data fetching function
get_cached_data <- function(data_type, fetch_function) {
  current_time <- Sys.time()
  cache_key <- paste0("last_", data_type, "_update")
  
  # Check if cache is valid
  if (is.null(data_cache[[data_type]]) || 
      is.null(data_cache[[cache_key]]) ||
      as.numeric(difftime(current_time, data_cache[[cache_key]], units = "secs")) > data_cache$cache_timeout) {
    
    # Fetch fresh data
    tryCatch({
      data_cache[[data_type]] <- fetch_function()
      data_cache[[cache_key]] <- current_time
      print(paste("Cache updated for:", data_type))
    }, error = function(e) {
      print(paste("Error fetching", data_type, ":", e$message))
    })
  }
  
  return(data_cache[[data_type]])
}

# Optimized data fetching functions
fetch_bandit_data <- function() {
  res <- GET(paste0(API_URL, "/bandit_simulation_results"))
  if (res$status_code == 200) {
    data <- fromJSON(rawToChar(res$content))
    return(data$data)
  }
  return(NULL)
}

fetch_dps_data <- function() {
  res <- GET(paste0(API_URL, "/user_dynamic_price_sensitivity_data"))
  if (res$status_code == 200) {
    data <- fromJSON(rawToChar(res$content))
    return(data$data)
  }
  return(NULL)
}

fetch_conversion_data <- function() {
  res <- GET(paste0(API_URL, "/conversion_probabilities_data"))
  if (res$status_code == 200) {
    data <- fromJSON(rawToChar(res$content))
    return(data$data)
  }
  return(NULL)
}

fetch_ranking_data <- function() {
  res <- POST(paste0(API_URL, "/rank"))
  if (res$status_code == 200) {
    data <- fromJSON(rawToChar(res$content))
    return(data)
  }
  return(NULL)
}

fetch_shapley_data <- function() {
  res <- POST(paste0(API_URL, "/calculate_shapley_values"))
  if (res$status_code == 200) {
    data <- fromJSON(rawToChar(res$content))
    return(data)
  }
  return(NULL)
}

fetch_policy_heatmap <- function() {
  tryCatch({
    print("[DEBUG] Fetching policy heatmap data...")
    res <- GET(paste0(API_URL, "/get_policy_heatmap"))
    print(paste("[DEBUG] Response status:", res$status_code))
    
    if (res$status_code == 200) {
      data <- fromJSON(rawToChar(res$content))
      print(paste("[DEBUG] Data fetched successfully, scenarios:", length(data$heatmap_data)))
      return(data)
    } else {
      print(paste("[ERROR] API returned status:", res$status_code))
      return(NULL)
    }
  }, error = function(e) {
    print(paste("[ERROR] Fetch policy heatmap error:", e$message))
    return(NULL)
  })
}

# --- UI Definition ---
ui <- dashboardPage(
  dashboardHeader(title = "trivago Strategic Ranking Simulator"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Strategic Levers", tabName = "strategic_levers", icon = icon("sliders-h")),
      menuItem("Optimization & Trade-offs", tabName = "optimization", icon = icon("balance-scale")),
      menuItem("Ecosystem Health", tabName = "ecosystem", icon = icon("heartbeat")),
      menuItem("Causal Impact (A/B Test)", tabName = "causal_impact", icon = icon("flask")),
      menuItem("Data Generation", tabName = "data_generation", icon = icon("database")),
      menuItem("Data Status", tabName = "data_status", icon = icon("folder-open"))
    )
  ),
  
  dashboardBody(
    useShinyjs(),
    
    # Custom CSS
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f8f9fa;
        }
        .box {
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-box {
          background: white;
          border-radius: 8px;
          padding: 15px;
          margin: 5px;
          text-align: center;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
          font-size: 24px;
          font-weight: bold;
          color: #2c3e50;
        }
        .metric-label {
          font-size: 12px;
          color: #7f8c8d;
          text-transform: uppercase;
        }
        .strategy-card {
          background: white;
          border-radius: 8px;
          padding: 15px;
          margin: 10px 0;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border-left: 4px solid #3498db;
        }
        .greedy-card { border-left-color: #e74c3c; }
        .user-first-card { border-left-color: #27ae60; }
        .stochastic-card { border-left-color: #f39c12; }
        .rl-card { border-left-color: #9b59b6; }
        
        /* MathJax improvements for better equation rendering */
        .MathJax_Display {
          overflow-x: auto;
          overflow-y: hidden;
          padding: 5px 0;
        }
        
        /* Responsive design for mathematical formulas */
        @media (max-width: 768px) {
          .math-container { 
            font-size: 12px !important; 
          }
          .MathJax_Display {
            font-size: 12px !important;
          }
        }
        
        /* Better spacing for mathematical content */
        .math-formula-container {
          margin-bottom: 20px;
          padding: 10px;
          background-color: #f8f9fa;
          border-radius: 5px;
        }
        
        /* Improved equation labels */
        .equation-label {
          font-weight: bold;
          color: #495057;
          margin-bottom: 8px;
          font-size: 13px;
        }
      "))
    ),
    
    # Add MathJax support for LaTeX rendering
    tags$head(
      tags$script(src = "https://polyfill.io/v3/polyfill.min.js?features=es6"),
      tags$script(id = "MathJax-script", async = TRUE, src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),
      tags$script("MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\\\(','\\\\)']]}});")
    ),

    tabItems(
      # --- TAB 1: STRATEGIC LEVERS ---
      tabItem(tabName = "strategic_levers",
        fluidRow(
          box(
            title = "Strategic Simulation Control", status = "primary", solidHeader = TRUE, width = 12,
            fluidRow(
              column(12,
                actionButton("run_simulation_btn", "Run Strategic Simulation", 
                           class = "btn-success btn-block", icon = icon("play"))
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Optimization Weights (α, β, γ)", status = "info", solidHeader = TRUE, width = 12,
            fluidRow(
              column(4,
                sliderInput("alpha_weight", "α - trivago Income Weight", 
                           min = 0, max = 1, value = 0.4, step = 0.1)
              ),
              column(4,
                sliderInput("beta_weight", "β - User Satisfaction Weight", 
                           min = 0, max = 1, value = 0.3, step = 0.1)
              ),
              column(4,
                sliderInput("gamma_weight", "γ - Partner Conversion Weight", 
                           min = 0, max = 1, value = 0.3, step = 0.1)
              )
            ),
            fluidRow(
              column(12,
                div(style = "text-align: center; padding: 10px;",
                    strong("Total Weight: "), 
                    textOutput("total_weight", inline = TRUE),
                    " (should equal 1.0)"
                )
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Strategic Policy Selection", status = "warning", solidHeader = TRUE, width = 12,
            fluidRow(
              column(6,
                actionButton("select_policy_btn", "Select Optimal Policy", 
                           class = "btn-warning", icon = icon("brain"))
              ),
              column(6,
                actionButton("train_rl_btn", "Train RL Agent", 
                           class = "btn-info", icon = icon("graduation-cap"))
              )
            ),
            fluidRow(
              column(12,
                verbatimTextOutput("policy_selection_output")
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Simulation Status", status = "success", solidHeader = TRUE, width = 12,
            verbatimTextOutput("simulation_status")
          )
        )
      ),
      
      # --- TAB 2: OPTIMIZATION & TRADE-OFFS ---
      tabItem(tabName = "optimization",
        fluidRow(
          box(
            title = "Pareto Frontier: Revenue vs. User Trust", status = "success", solidHeader = TRUE, width = 6,
            plotlyOutput("pareto_frontier_plot")
          ),
          box(
            title = "Learned RL Policy Table", status = "info", solidHeader = TRUE, width = 6,
            textOutput("test_connection"),
            DTOutput("policy_table")
          )
        ),
        fluidRow(
          box(
            title = "Optimization Results", status = "primary", solidHeader = TRUE, width = 12,
            fluidRow(
              column(6,
                actionButton("run_optimization_btn", "Run Optimization", 
                           class = "btn-primary", icon = icon("cogs"))
              ),
              column(4,
                actionButton("run_deterministic_btn", "Run Deterministic Optimization", 
                           class = "btn-warning", icon = icon("calculator"))
              ),
              column(4,
                actionButton("run_stochastic_btn", "Run Stochastic Optimization", 
                           class = "btn-info", icon = icon("dice"))
              )
            ),
            br(), br(),
            DT::dataTableOutput("optimization_results_table")
          )
        ),
        fluidRow(
          box(
            title = "Mathematical Foundation", status = "warning", solidHeader = TRUE, width = 12,
            withMathJax(
              div(style = "font-family: 'Times New Roman', serif; font-size: 14px; line-height: 1.8;",
                tags$div(style = "margin-bottom: 20px; overflow-x: auto;",
                  helpText("Multi-Objective Function:"),
                  "$$\\text{Total Objective} = \\alpha \\cdot \\text{Trivago Income} + \\beta \\cdot \\text{User Satisfaction} + \\gamma \\cdot \\text{Partner Conversion Value}$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Budget Constraint:"),
                  "$$\\sum_{i \\in P} \\text{cost\\_per\\_click\\_bid}_i \\leq \\text{remaining\\_budget}_P \\quad \\forall P \\in \\text{Partners}$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Position-based CTR:"),
                  "$$\\text{CTR}(\\text{position}) = \\frac{1}{1 + 0.5 \\cdot \\text{position}}$$"
                )
              )
            )
          )
        )
      ),
      
      # --- TAB 3: ECOSYSTEM HEALTH ---
      tabItem(tabName = "ecosystem",
        fluidRow(
          box(
            title = "Partner Budget Consumption", status = "warning", solidHeader = TRUE, width = 6,
            plotlyOutput("budget_consumption_plot")
          ),
          box(
            title = "Partner Contribution (Shapley Values)", status = "success", solidHeader = TRUE, width = 6,
            plotlyOutput("shapley_values_plot")
          )
        ),
        fluidRow(
          box(
            title = "Ecosystem Metrics", status = "info", solidHeader = TRUE, width = 12,
            fluidRow(
              column(3,
                div(class = "metric-box",
                  div(class = "metric-value", textOutput("total_revenue")),
                  div(class = "metric-label", "Total Revenue")
                )
              ),
              column(3,
                div(class = "metric-box",
                  div(class = "metric-value", textOutput("avg_satisfaction")),
                  div(class = "metric-label", "Avg Satisfaction")
                )
              ),
              column(3,
                div(class = "metric-box",
                  div(class = "metric-value", textOutput("conversion_rate")),
                  div(class = "metric-label", "Conversion Rate")
                )
              ),
              column(3,
                div(class = "metric-box",
                  div(class = "metric-value", textOutput("budget_utilization")),
                  div(class = "metric-label", "Budget Utilization")
                )
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Partner Performance Details", status = "primary", solidHeader = TRUE, width = 12,
            actionButton("calculate_shapley_btn", "Calculate Shapley Values", 
                       class = "btn-primary", icon = icon("calculator")),
            br(), br(),
            DT::dataTableOutput("partner_performance_table")
          )
        )
      ),
      
      # --- TAB 4: CAUSAL IMPACT (A/B TEST) ---
      tabItem(tabName = "causal_impact",
        fluidRow(
          box(
            title = "A/B Test Configuration", status = "primary", solidHeader = TRUE, width = 12,
            fluidRow(
              column(4,
                selectInput("control_strategy", "Control Strategy:", 
                           choices = c("Greedy", "User-First"), selected = "Greedy")
              ),
              column(4,
                selectInput("treatment_strategy", "Treatment Strategy:", 
                           choices = c("LP-Optimized", "RL Policy"), selected = "LP-Optimized")
              ),
              column(4,
                numericInput("test_duration", "Test Duration (days):", value = 30, min = 7, max = 90)
              )
            ),
            fluidRow(
              column(12,
                actionButton("run_ab_test_btn", "Run A/B Test", 
                           class = "btn-success btn-block", icon = icon("flask"))
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Causal Impact Results", status = "success", solidHeader = TRUE, width = 12,
            DT::dataTableOutput("ab_test_results_table")
          )
        ),
        fluidRow(
          box(
            title = "Statistical Significance", status = "info", solidHeader = TRUE, width = 6,
            plotlyOutput("significance_plot")
          ),
          box(
            title = "Treatment Effect Timeline", status = "warning", solidHeader = TRUE, width = 6,
            plotlyOutput("treatment_effect_plot")
          )
        ),
        fluidRow(
          box(
            title = "Causal Inference Methodology", status = "warning", solidHeader = TRUE, width = 12,
            withMathJax(
              div(style = "font-family: 'Times New Roman', serif; font-size: 14px; line-height: 1.8;",
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Treatment Effect:"),
                  "$$\\tau = E[Y(1) - Y(0)] = E[Y|T=1] - E[Y|T=0]$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Uplift Calculation:"),
                  "$$\\text{Uplift} = \\frac{\\text{Treatment} - \\text{Control}}{\\text{Control}} \\times 100\\%$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Statistical Significance:"),
                  "$$p\\text{-value} = P(|Z| > |z_{obs}|) \\text{ where } Z \\sim N(0,1)$$"
                )
              )
            )
          )
        )
      ),
      
      # --- TAB 5: DATA GENERATION ---
      tabItem(tabName = "data_generation",
        fluidRow(
          box(
            title = "Data Generation Parameters", status = "primary", solidHeader = TRUE, width = 12,
            fluidRow(
              column(3,
                numericInput("num_users_gen", "Number of Users:", value = 80, min = 1, max = 100)
              ),
              column(3,
                numericInput("num_hotels_gen", "Hotels per Destination:", value = 10, min = 1, max = 20)
              ),
              column(3,
                numericInput("num_partners_gen", "Partners per Hotel:", value = 5, min = 1, max = 10)
              ),
              column(3,
                numericInput("min_users_per_destination_gen", "Min Users per Destination:", value = 8, min = 1, max = 20)
              )
            ),
            fluidRow(
              column(3,
                numericInput("days_to_go_gen", "Days to Go (target):", value = 30, min = 1, max = 365)
              ),
              column(3,
                numericInput("days_var_gen", "Days Variance:", value = 5, min = 1, max = 30)
              ),
              column(6,
                actionButton("generate_data_btn", "Generate Data", 
                           class = "btn-success btn-block", icon = icon("database"))
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Generated Data", status = "success", solidHeader = TRUE, width = 12,
            tabsetPanel(
              tabPanel("Bandit Results", DT::dataTableOutput("bandit_table")),
              tabPanel("User DPS", DT::dataTableOutput("dps_table")),
              tabPanel("Conversion Probs", DT::dataTableOutput("conversion_table"))
            )
          )
        )
      ),
      
      # --- TAB 6: DATA STATUS ---
      tabItem(tabName = "data_status",
        fluidRow(
          box(
            title = "Data Files Status", status = "info", solidHeader = TRUE, width = 12,
            actionButton("refresh_data_status_btn", "Refresh Data Status", 
                       class = "btn-primary", icon = icon("refresh")),
            br(), br(),
            DT::dataTableOutput("data_status_table")
          )
        ),
        fluidRow(
          box(
            title = "Data Summary", status = "success", solidHeader = TRUE, width = 6,
            div(class = "metric-box",
              div(class = "metric-value", textOutput("total_data_files")),
              div(class = "metric-label", "Total Files")
            ),
            div(class = "metric-box",
              div(class = "metric-value", textOutput("total_data_size")),
              div(class = "metric-label", "Total Size (MB)")
            )
          ),
          box(
            title = "Data Directory Info", status = "warning", solidHeader = TRUE, width = 6,
            verbatimTextOutput("data_directory_info")
          )
        )
      )
    )
  )
)

# --- Server Logic ---
server <- function(input, output, session) {
  
  # Reactive values for UI state management
  rv <- reactiveValues(
    simulation_message = NULL,
    policy_selection_result = NULL,
    optimization_results = NULL,
    ab_test_results = NULL,
    refresh_counter = 0
  )
  
  # --- STRATEGIC LEVERS TAB ---
  
  # Total weight calculation
  output$total_weight <- renderText({
    total <- input$alpha_weight + input$beta_weight + input$gamma_weight
    if (abs(total - 1.0) > 0.01) {
      paste0(total, " ⚠️")
    } else {
      paste0(total, " ✅")
    }
  })
  
  # Run strategic simulation
  observeEvent(input$run_simulation_btn, {
    tryCatch({
      showNotification("Running strategic simulation...", type = "message")
      
      # Step 1: Sample data using parameters from Data Generation tab
      res1 <- POST(paste0(API_URL, "/sample_offers_for_users"), 
                 query = list(
                   num_users = input$num_users_gen,
                   num_hotels = input$num_hotels_gen,
                   num_partners = input$num_partners_gen,
                   days_to_go = input$days_to_go_gen,
                   days_var = input$days_var_gen,
                   min_users_per_destination = input$min_users_per_destination_gen
                 ))
      
      if (res1$status_code != 200) {
        showNotification("Error in data sampling", type = "error")
        return()
      }
      
      # Step 2: Run optimization with current weights
      res2 <- POST(paste0(API_URL, "/rank"), 
                 query = list(
                   alpha = input$alpha_weight,
                   beta = input$beta_weight,
                   gamma = input$gamma_weight,
                   num_positions = 10
                 ))
      
      if (res2$status_code != 200) {
        showNotification("Error in optimization", type = "error")
        return()
      }
      
      optimization_data <- fromJSON(rawToChar(res2$content))
      rv$optimization_results <- optimization_data
      
      # Step 3: Generate CSVs
      POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"))
      POST(paste0(API_URL, "/conversion_probabilities_csv"))
      
      rv$simulation_message <- paste0(
        "✅ Strategic simulation completed!\n",
        "📊 Optimization Results:\n",
        "• Trivago Income: $", round(optimization_data$objectives$trivago_income, 2), "\n",
        "• User Satisfaction: ", round(optimization_data$objectives$user_satisfaction, 2), "\n",
        "• Partner Conversion Value: ", round(optimization_data$objectives$partner_conversion_value, 2), "\n",
        "• Total Objective: ", round(optimization_data$objectives$total_objective, 2), "\n\n",
        "⚖️ Weights Used: α=", input$alpha_weight, ", β=", input$beta_weight, ", γ=", input$gamma_weight
      )
      
      showNotification("Strategic simulation completed successfully!", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Select optimal policy
  observeEvent(input$select_policy_btn, {
    tryCatch({
      showNotification("Selecting optimal policy...", type = "message")
      
      res <- POST(paste0(API_URL, "/select_strategic_policy"))
      
      if (res$status_code == 200) {
        policy_data <- fromJSON(rawToChar(res$content))
        rv$policy_selection_result <- policy_data
        
        output$policy_selection_output <- renderText({
          paste0(
            "🎯 Selected Policy: ", policy_data$selected_policy$policy_name, "\n",
            "⚖️ Weights: α=", policy_data$selected_policy$weights$alpha, 
            ", β=", policy_data$selected_policy$weights$beta,
            ", γ=", policy_data$selected_policy$weights$gamma, "\n",
            "🧠 Epsilon: ", round(policy_data$epsilon, 4), "\n",
            "📊 Market State:\n",
            "• Demand: ", policy_data$market_state$market_demand, "\n",
            "• Days to Go: ", round(policy_data$market_state$days_to_go, 1), "\n",
            "• Competition: ", policy_data$market_state$competition_density
          )
        })
        
        showNotification("Optimal policy selected!", type = "message")
      } else {
        showNotification("Error selecting policy", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Train RL agent
  observeEvent(input$train_rl_btn, {
    tryCatch({
      showNotification("Training RL agent...", type = "message")
      
      res <- POST(paste0(API_URL, "/train_rl_agent"))
      
      if (res$status_code == 200) {
        training_data <- fromJSON(rawToChar(res$content))
        
        showNotification(paste0(
          "RL Agent trained! Policy: ", training_data$training_result$policy_name,
          ", Reward: ", round(training_data$training_result$reward, 4)
        ), type = "message")
      } else {
        showNotification("Error training RL agent", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Simulation status
  output$simulation_status <- renderText({
    if (is.null(rv$simulation_message)) {
      return("Click 'Run Strategic Simulation' to start...")
    }
    rv$simulation_message
  })
  
  # --- OPTIMIZATION & TRADE-OFFS TAB ---
  
  # Pareto frontier plot
  output$pareto_frontier_plot <- renderPlotly({
    if (is.null(rv$optimization_results)) {
      return(plot_ly() %>% 
               add_annotations(text = "Run optimization to see Pareto frontier", 
                             showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
    }
    
    # Generate Pareto frontier data by varying weights
    pareto_data <- data.frame()
    
    for (alpha in seq(0, 1, 0.1)) {
      for (beta in seq(0, 1 - alpha, 0.1)) {
        gamma <- 1 - alpha - beta
        if (gamma >= 0) {
          pareto_data <- rbind(pareto_data, data.frame(
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            revenue = alpha * 1000,  # Simplified
            trust = beta * 10        # Simplified
          ))
        }
      }
    }
    
    plot_ly(pareto_data, x = ~revenue, y = ~trust, 
            type = 'scatter', mode = 'markers',
            marker = list(size = 8, color = ~alpha, colorscale = 'Viridis'),
            text = ~paste("α:", alpha, "<br>β:", beta, "<br>γ:", gamma),
            hoverinfo = 'text') %>%
      layout(
        title = "Pareto Frontier: Revenue vs. User Trust",
        xaxis = list(title = "Expected Revenue ($)"),
        yaxis = list(title = "User Trust Score"),
        showlegend = FALSE
      )
  })
  
  # Test connection to backend
  output$test_connection <- renderText({
    tryCatch({
      res <- GET(paste0(API_URL, "/"))
      if (res$status_code == 200) {
        "Backend connection: OK"
      } else {
        paste("Backend connection failed:", res$status_code)
      }
    }, error = function(e) {
      paste("Backend connection error:", e$message)
    })
  })
  
  # Policy table
  output$policy_table <- renderDT({
    tryCatch({
      print("[DEBUG] Policy table rendering...")
      heatmap_data <- get_cached_data("policy_heatmap", fetch_policy_heatmap)
      print(paste("[DEBUG] Policy data received:", !is.null(heatmap_data)))
      
      if (is.null(heatmap_data) || "error" %in% names(heatmap_data)) {
        print("[DEBUG] No policy data available")
        return(data.frame(
          Message = "No policy data available",
          stringsAsFactors = FALSE
        ))
      }
      
      print(paste("[DEBUG] Heatmap data structure:", class(heatmap_data$heatmap_data)))
      print(paste("[DEBUG] Number of scenarios:", length(heatmap_data$heatmap_data)))
      
      # Convert to data frame with robust extraction
      if (!is.data.frame(heatmap_data$heatmap_data)) {
        print("[DEBUG] Converting list to data frame...")
        
        # Improved extraction with error handling
        extract_value <- function(x, field) {
          tryCatch({
            val <- x[[field]]
            
            # Handle different data types
            if (is.list(val)) {
              if (length(val) > 0) {
                # If it's a named list, take the first element
                if (!is.null(names(val))) val <- val[[1]]
                # Otherwise, return as is
              } else {
                val <- NA
              }
            }
            
            # Convert to appropriate type
            if (field == "best_policy") {
              return(as.character(val))
            } else {
              return(as.numeric(val))
            }
          }, error = function(e) {
            print(paste("[WARNING] Error extracting", field, ":", e$message))
            return(NA)
          })
        }
        
        # Extract data with improved conversion
        competition_density <- sapply(heatmap_data$heatmap_data, extract_value, "competition_density")
        market_demand <- sapply(heatmap_data$heatmap_data, extract_value, "market_demand")
        days_to_go <- sapply(heatmap_data$heatmap_data, extract_value, "days_to_go")
        best_policy <- sapply(heatmap_data$heatmap_data, extract_value, "best_policy")
        
        # Special handling for q_values - show individual policy Q-values
        q_values_processed <- sapply(heatmap_data$heatmap_data, function(x) {
          tryCatch({
            qvals <- x$q_values
            
            # Handle both vectors and nested lists
            if (is.list(qvals)) {
              # Deep unlist to handle nested lists
              qvals <- unlist(qvals, recursive = TRUE)
            }
            
            # Ensure we have a numeric vector
            qvals <- as.numeric(qvals)
            
            if (length(qvals) > 0 && is.numeric(qvals)) {
              # Round each Q-value to 3 decimal places and join with commas
              rounded_qvals <- round(qvals, 3)
              paste(rounded_qvals, collapse = ", ")
            } else {
              NA
            }
          }, error = function(e) {
            print(paste("[WARNING] Error processing q_values:", e$message))
            return(NA)
          })
        })
        
        df_full <- data.frame(
          Competition_Density = competition_density,
          Market_Demand = market_demand,
          Days_to_Go = days_to_go,
          Q_Values = q_values_processed,
          Best_Policy = best_policy,
          stringsAsFactors = FALSE
        )
      } else {
        print("[DEBUG] Data is already a data frame")
        df_full <- heatmap_data$heatmap_data
        
        # Ensure proper column names and types
        colnames(df_full) <- c("Competition_Density", "Market_Demand", "Days_to_Go", "Q_Values", "Best_Policy")
        df_full$Competition_Density <- as.numeric(df_full$Competition_Density)
        df_full$Market_Demand <- as.numeric(df_full$Market_Demand)
        df_full$Days_to_Go <- as.numeric(df_full$Days_to_Go)
        df_full$Q_Values <- as.character(df_full$Q_Values)
        df_full$Best_Policy <- as.character(df_full$Best_Policy)
      }
      
      print(paste("[DEBUG] Data frame created with", nrow(df_full), "rows"))
      
      # Sort data
      df_full <- df_full[order(
        df_full$Competition_Density,
        -df_full$Days_to_Go,
        df_full$Market_Demand
      ), ]
      
      print("[DEBUG] First 6 rows of data:")
      print(head(df_full))
      
      return(df_full)
      
    }, error = function(e) {
      print(paste("[ERROR] Policy table error:", e$message))
      data.frame(
        Error = paste("Error:", e$message),
        stringsAsFactors = FALSE
      )
    })
  }, 
  options = list(
    pageLength = 10,
    scrollX = TRUE,
    dom = 'ftip'
  ),
  rownames = FALSE
  )
  
  # Run optimization
  observeEvent(input$run_optimization_btn, {
    tryCatch({
      showNotification("Running optimization...", type = "message")
      
      res <- POST(paste0(API_URL, "/rank"), 
                 query = list(
                   alpha = input$alpha_weight,
                   beta = input$beta_weight,
                   gamma = input$gamma_weight,
                   num_positions = 10
                 ))
      
      if (res$status_code == 200) {
        optimization_data <- fromJSON(rawToChar(res$content))
        rv$optimization_results <- optimization_data
        showNotification("Optimization completed!", type = "message")
      } else {
        showNotification("Error in optimization", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Run deterministic optimization
  observeEvent(input$run_deterministic_btn, {
    tryCatch({
      showNotification("Running deterministic optimization...", type = "message")
      
      res <- POST(paste0(API_URL, "/run_deterministic_optimization"))
      
      if (res$status_code == 200) {
        deterministic_data <- fromJSON(rawToChar(res$content))
        showNotification("Deterministic optimization completed!", type = "message")
        
        # Update the reactive value so the table can display the results
        rv$optimization_results <- deterministic_data
        
        # Show results in a simple format
        if (!is.null(deterministic_data$results)) {
          showNotification(paste("Results saved to /data folder. Total expected income:", 
                               round(deterministic_data$results$json$optimization_results$total_expected_income, 2)), 
                         type = "message")
        }
      } else {
        showNotification("Error in deterministic optimization", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Run stochastic optimization
  observeEvent(input$run_stochastic_btn, {
    tryCatch({
      showNotification("Running stochastic optimization...", type = "message")
      
      res <- POST(paste0(API_URL, "/run_stochastic_optimization"))
      
      if (res$status_code == 200) {
        stochastic_data <- fromJSON(rawToChar(res$content))
        showNotification("Stochastic optimization completed!", type = "message")
        
        # Update the reactive value so the table can display the results
        rv$optimization_results <- stochastic_data
        
        # Show results in a simple format
        if (!is.null(stochastic_data$results)) {
          showNotification(paste("Results saved to /data folder. Total expected value:", 
                               round(stochastic_data$results$optimization_results$total_expected_value, 2)), 
                         type = "message")
        }
      } else {
        showNotification("Error in stochastic optimization", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
    # Helper function to safely extract values from nested structures
  safe_extract <- function(obj, field, default = NULL) {
    if (is.null(obj)) return(default)
    if (!is.list(obj)) return(default)
    if (is.null(obj[[field]])) return(default)
    return(obj[[field]])
  }
  
  # Helper function to safely access list elements
  safe_access <- function(obj, field, default = NULL) {
    if (is.null(obj)) return(default)
    if (!is.list(obj)) return(default)
    if (is.null(obj[[field]])) return(default)
    return(obj[[field]])
  }
  
  # Optimization results table
  output$optimization_results_table <- DT::renderDataTable({
    tryCatch({
      if (is.null(rv$optimization_results)) {
        return(data.frame(Message = "Run optimization to see results"))
      }
      
      # Debug: print the structure of optimization results
      print("Optimization results structure:")
      print(str(rv$optimization_results))
      
      # Check if this is deterministic/stochastic optimization results
      if (!is.null(safe_access(rv$optimization_results, "results")) && 
          !is.null(safe_access(rv$optimization_results$results, "json"))) {
        
        # Handle deterministic/stochastic optimization results
        opt_results <- rv$optimization_results$results$json
        
        if (!is.null(safe_access(opt_results, "optimization_results"))) {
          opt_data <- opt_results$optimization_results
          
          # Deterministic optimization
          if (!is.null(safe_access(opt_data, "user_rankings"))) {
            user_rankings <- opt_data$user_rankings
            offers <- safe_access(opt_results$original_data, "offers", list())
            
            # Create ranking table from user rankings
            ranking_data <- list()
            
            # Handle the case where user_rankings is a nested array
            if (is.list(user_rankings) && length(user_rankings) > 0) {
              for (user_idx in seq_along(user_rankings)) {
                ranking <- user_rankings[[user_idx]]
                # ranking is now a vector of offer IDs
                if (is.numeric(ranking) || is.list(ranking)) {
                  for (rank in seq_along(ranking)) {
                    offer_id <- ranking[rank]
                    if (is.numeric(offer_id) && offer_id > 0 && offer_id <= length(offers)) {
                      offer <- offers[[offer_id]]
                      # Check if offer is a list before accessing with $
                      if (is.list(offer)) {
                        ranking_data[[length(ranking_data) + 1]] <- list(
                          Position = rank,
                          User = paste0("User ", user_idx),
                          Offer_ID = safe_extract(offer, "offer_id", offer_id),
                          Conversion_Prob = round(safe_extract(offer, "conversion_probability", 0), 3),
                          Bid_Amount = round(safe_extract(offer, "bid_amount", 0), 3),
                          Hotel_Type = safe_extract(offer, "hotel_type", "unknown"),
                          Price_Level = safe_extract(offer, "price_level", "unknown")
                        )
                      }
                    }
                  }
                }
              }
            }
            
            if (length(ranking_data) > 0) {
              df <- do.call(rbind, lapply(ranking_data, function(x) {
                data.frame(
                  Position = x$Position,
                  User = x$User,
                  Offer_ID = x$Offer_ID,
                  Conversion_Prob = x$Conversion_Prob,
                  Bid_Amount = x$Bid_Amount,
                  Hotel_Type = x$Hotel_Type,
                  Price_Level = x$Price_Level,
                  stringsAsFactors = FALSE
                )
              }))
              
              return(DT::datatable(df, 
                                  options = list(pageLength = 10, scrollX = TRUE),
                                  rownames = FALSE))
            } else {
              return(data.frame(Message = "No ranking data generated from deterministic optimization"))
            }
          }
          
          # Stochastic optimization
          if (!is.null(safe_access(opt_data, "selected_offers"))) {
            selected_offers <- opt_data$selected_offers
            
            if (length(selected_offers) > 0) {
              df <- do.call(rbind, lapply(selected_offers, function(offer) {
                # Check if offer is a list before accessing with $
                if (is.list(offer)) {
                  data.frame(
                    Offer_ID = safe_extract(offer, "offer_id", "unknown"),
                    Conversion_Prob = round(safe_extract(offer, "conversion_probability", 0), 3),
                    Revenue = round(safe_extract(offer, "revenue", 0), 3),
                    Trust_Score = round(safe_extract(offer, "trust_score", 0), 3),
                    Price_Consistency = round(safe_extract(offer, "price_consistency", 0), 3),
                    stringsAsFactors = FALSE
                  )
                } else {
                  # Fallback for non-list offers
                  data.frame(
                    Offer_ID = "unknown",
                    Conversion_Prob = 0,
                    Revenue = 0,
                    Trust_Score = 0,
                    Price_Consistency = 0,
                    stringsAsFactors = FALSE
                  )
                }
              }))
              
              return(DT::datatable(df, 
                                  options = list(pageLength = 10, scrollX = TRUE),
                                  rownames = FALSE))
            } else {
              return(data.frame(Message = "No offers selected in stochastic optimization"))
            }
          }
        }
      }
      
      # Handle original optimization results structure
      ranking_data <- safe_access(rv$optimization_results, "ranking")
      if (is.null(ranking_data)) {
        return(data.frame(Message = "No ranking data available"))
      }
      
      df <- do.call(rbind, lapply(ranking_data, function(x) {
        if (is.list(x)) {
          data.frame(
            Position = safe_extract(x, "position", "N/A"),
            Hotel = safe_extract(x, "hotel_id", "N/A"),
            Partner = safe_extract(x, "partner_name", "N/A"),
            Price = paste0("$", safe_extract(x, "price_per_night", 0)),
            Commission = paste0(round(safe_extract(x, "commission_rate", 0) * 100, 1), "%"),
            Satisfaction = round(safe_extract(x, "user_satisfaction_score", 0), 3),
            Conversion = round(safe_extract(x, "conversion_probability", 0), 3),
            stringsAsFactors = FALSE
          )
        } else {
          # Fallback for non-list items
          data.frame(
            Position = "N/A",
            Hotel = "N/A", 
            Partner = "N/A",
            Price = "$0",
            Commission = "0%",
            Satisfaction = 0,
            Conversion = 0,
            stringsAsFactors = FALSE
          )
        }
      }))
      
      DT::datatable(df, 
                    options = list(pageLength = 10, scrollX = TRUE),
                    rownames = FALSE)
      
    }, error = function(e) {
      print(paste("Error in optimization results table:", e$message))
      print(paste("Error call:", deparse(e$call)))
      return(data.frame(Error = paste("Error displaying results:", e$message)))
    })
  })
  
  # --- ECOSYSTEM HEALTH TAB ---
  
  # Budget consumption plot
  output$budget_consumption_plot <- renderPlotly({
    ranking_data <- rv$optimization_results
    if (is.null(ranking_data) || is.null(ranking_data$ranking)) {
      return(plot_ly() %>% 
               add_annotations(text = "Run optimization to see budget data", 
                             showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
    }
    
    # Aggregate budget data by partner
    budget_data <- do.call(rbind, lapply(ranking_data$ranking, function(x) {
      data.frame(
        Partner = x$partner_name,
        Remaining = x$remaining_budget,
        Used = x$partner_marketing_budget - x$remaining_budget
      )
    }))
    
    budget_summary <- aggregate(. ~ Partner, budget_data, sum)
    
    plot_ly(budget_summary, x = ~Partner, y = ~Remaining, 
            type = 'bar', name = 'Remaining Budget',
            marker = list(color = '#2ecc71')) %>%
      add_trace(y = ~Used, name = 'Used Budget',
                marker = list(color = '#e74c3c')) %>%
      layout(
        title = "Partner Budget Consumption",
        xaxis = list(title = "Partner"),
        yaxis = list(title = "Budget ($)"),
        barmode = 'stack'
      )
  })
  
  # Shapley values plot
  output$shapley_values_plot <- renderPlotly({
    shapley_data <- get_cached_data("shapley_data", fetch_shapley_data)
    
    if (is.null(shapley_data) || "error" %in% names(shapley_data)) {
      return(plot_ly() %>% 
               add_annotations(text = "Calculate Shapley values to see data", 
                             showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
    }
    
    df <- do.call(rbind, shapley_data$shapley_values)
    
    plot_ly(df, x = ~partner_name, y = ~contribution_percentage, 
            type = 'bar',
            marker = list(color = '#3498db')) %>%
      layout(
        title = "Partner Contribution (Shapley Values)",
        xaxis = list(title = "Partner"),
        yaxis = list(title = "Contribution (%)")
      )
  })
  
  # Calculate Shapley values
  observeEvent(input$calculate_shapley_btn, {
    tryCatch({
      showNotification("Calculating Shapley values...", type = "message")
      
      shapley_data <- fetch_shapley_data()
      if (!is.null(shapley_data)) {
        data_cache$shapley_data <- shapley_data
        data_cache$last_shapley_update <- Sys.time()
        showNotification("Shapley values calculated!", type = "message")
      } else {
        showNotification("Error calculating Shapley values", type = "error")
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Ecosystem metrics
  output$total_revenue <- renderText({
    if (is.null(rv$optimization_results)) return("N/A")
    paste0("$", round(rv$optimization_results$objectives$trivago_income, 0))
  })
  
  output$avg_satisfaction <- renderText({
    if (is.null(rv$optimization_results)) return("N/A")
    round(rv$optimization_results$objectives$user_satisfaction, 2)
  })
  
  output$conversion_rate <- renderText({
    if (is.null(rv$optimization_results)) return("N/A")
    paste0(round(rv$optimization_results$objectives$partner_conversion_value * 100, 1), "%")
  })
  
  output$budget_utilization <- renderText({
    if (is.null(rv$optimization_results)) return("N/A")
    "75%"  # Simplified
  })
  
  # Partner performance table
  output$partner_performance_table <- DT::renderDataTable({
    shapley_data <- get_cached_data("shapley_data", fetch_shapley_data)
    
    if (is.null(shapley_data) || "error" %in% names(shapley_data)) {
      return(data.frame(Message = "Calculate Shapley values to see partner performance"))
    }
    
    df <- do.call(rbind, shapley_data$shapley_values)
    
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("avg_commission_rate", "expected_revenue", "shapley_value"), digits = 4) %>%
      DT::formatRound(columns = c("contribution_percentage"), digits = 2)
  })
  
  # --- CAUSAL IMPACT TAB ---
  
  # Run A/B test
  observeEvent(input$run_ab_test_btn, {
    tryCatch({
      showNotification("Running A/B test...", type = "message")
      
      # Simulate A/B test results
      control_results <- list(
        revenue = 8500,
        conversion_rate = 0.12,
        user_trust = 7.8
      )
      
      treatment_results <- list(
        revenue = 9200,
        conversion_rate = 0.14,
        user_trust = 8.2
      )
      
      # Calculate uplift and significance
      revenue_uplift <- (treatment_results$revenue - control_results$revenue) / control_results$revenue * 100
      conversion_uplift <- (treatment_results$conversion_rate - control_results$conversion_rate) / control_results$conversion_rate * 100
      trust_uplift <- (treatment_results$user_trust - control_results$user_trust) / control_results$user_trust * 100
      
      # Simplified p-value calculation (in practice, would use proper statistical test)
      p_values <- c(0.023, 0.045, 0.012)  # Simulated p-values
      
      ab_test_results <- data.frame(
        Metric = c("Revenue", "Conversion Rate", "User Trust"),
        Control = c(control_results$revenue, control_results$conversion_rate, control_results$user_trust),
        Treatment = c(treatment_results$revenue, treatment_results$conversion_rate, treatment_results$user_trust),
        Uplift = c(revenue_uplift, conversion_uplift, trust_uplift),
        P_Value = p_values,
        Significant = p_values < 0.05
      )
      
      rv$ab_test_results <- ab_test_results
      
      showNotification("A/B test completed!", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # A/B test results table
  output$ab_test_results_table <- DT::renderDataTable({
    if (is.null(rv$ab_test_results)) {
      return(data.frame(Message = "Run A/B test to see results"))
    }
    
    DT::datatable(rv$ab_test_results, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("Control", "Treatment", "Uplift"), digits = 2) %>%
      DT::formatRound(columns = c("P_Value"), digits = 4) %>%
      DT::formatStyle("Significant", 
                     backgroundColor = styleEqual(c(TRUE, FALSE), c("#d4edda", "#f8d7da")))
  })
  
  # Significance plot
  output$significance_plot <- renderPlotly({
    if (is.null(rv$ab_test_results)) {
      return(plot_ly() %>% 
               add_annotations(text = "Run A/B test to see significance plot", 
                             showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
    }
    
    plot_ly(rv$ab_test_results, x = ~Metric, y = ~P_Value, 
            type = 'bar',
            marker = list(color = ifelse(rv$ab_test_results$P_Value < 0.05, '#2ecc71', '#e74c3c'))) %>%
      add_hline(y = 0.05, line = list(dash = "dash", color = "red")) %>%
      layout(
        title = "Statistical Significance (p-values)",
        xaxis = list(title = "Metric"),
        yaxis = list(title = "P-Value", range = c(0, 0.1))
      )
  })
  
  # Treatment effect plot
  output$treatment_effect_plot <- renderPlotly({
    if (is.null(rv$ab_test_results)) {
      return(plot_ly() %>% 
               add_annotations(text = "Run A/B test to see treatment effect", 
                             showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
    }
    
    plot_ly(rv$ab_test_results, x = ~Metric, y = ~Uplift, 
            type = 'bar',
            marker = list(color = ifelse(rv$ab_test_results$Uplift > 0, '#2ecc71', '#e74c3c'))) %>%
      layout(
        title = "Treatment Effect (Uplift %)",
        xaxis = list(title = "Metric"),
        yaxis = list(title = "Uplift (%)")
      )
  })
  
  # --- DATA GENERATION TAB (Legacy) ---
  
  # Generate data
  observeEvent(input$generate_data_btn, {
    tryCatch({
      showNotification("Generating data...", type = "message")
      
      res1 <- POST(paste0(API_URL, "/sample_offers_for_users"), 
                 query = list(
                   num_users = input$num_users_gen,
                   num_hotels = input$num_hotels_gen,
                   num_partners = input$num_partners_gen,
                   days_to_go = input$days_to_go_gen,
                   days_var = input$days_var_gen,
                   min_users_per_destination = input$min_users_per_destination_gen
                 ))
      
      if (res1$status_code != 200) {
        showNotification("Error in data generation", type = "error")
        return()
      }
      
      # Generate CSVs
      POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"))
      POST(paste0(API_URL, "/conversion_probabilities_csv"))
      
      showNotification("Data generated successfully!", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Data tables
  output$bandit_table <- DT::renderDataTable({
    bandit_data <- get_cached_data("bandit_data", fetch_bandit_data)
    
    if (is.null(bandit_data) || length(bandit_data) == 0) {
      return(data.frame(Message = "No bandit data available"))
    }
    
    df <- as.data.frame(bandit_data)
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("probability_of_click", "true_click_prob", "preference_score"), digits = 4)
  })
  
  output$dps_table <- DT::renderDataTable({
    dps_data <- get_cached_data("dps_data", fetch_dps_data)
    
    if (is.null(dps_data) || length(dps_data) == 0) {
      return(data.frame(Message = "No DPS data available"))
    }
    
    df <- as.data.frame(dps_data)
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("base_price_sensitivity", "dynamic_price_sensitivity"), digits = 4)
  })
  
  output$conversion_table <- DT::renderDataTable({
    conversion_data <- get_cached_data("conversion_data", fetch_conversion_data)
    
    if (is.null(conversion_data) || length(conversion_data) == 0) {
      return(data.frame(Message = "No conversion data available"))
    }
    
    df <- as.data.frame(conversion_data)
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("conversion_probability"), digits = 4)
  })
  
  # --- DATA STATUS TAB ---
  
  # Reactive value for data status
  data_status_rv <- reactiveVal(NULL)
  
  # Fetch data status function
  fetch_data_status <- function() {
    tryCatch({
      res <- GET(paste0(API_URL, "/data_status"))
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        return(data)
      }
      return(NULL)
    }, error = function(e) {
      print(paste("[ERROR] Fetch data status error:", e$message))
      return(NULL)
    })
  }
  
  # Refresh data status
  observeEvent(input$refresh_data_status_btn, {
    tryCatch({
      showNotification("Refreshing data status...", type = "message")
      status_data <- fetch_data_status()
      if (!is.null(status_data)) {
        data_status_rv(status_data)
        showNotification("Data status refreshed!", type = "message")
      } else {
        showNotification("Error fetching data status", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Data status table
  output$data_status_table <- DT::renderDataTable({
    status_data <- data_status_rv()
    if (is.null(status_data)) {
      return(data.frame(Message = "Click 'Refresh Data Status' to load data file information"))
    }
    
    # Convert files info to data frame
    files_df <- do.call(rbind, lapply(names(status_data$files), function(filename) {
      file_info <- status_data$files[[filename]]
      data.frame(
        File = filename,
        Exists = ifelse(file_info$exists, "✅ Yes", "❌ No"),
        Size_MB = file_info$size_mb,
        Last_Modified = ifelse(is.null(file_info$last_modified), "N/A", file_info$last_modified),
        stringsAsFactors = FALSE
      )
    }))
    
    DT::datatable(files_df, 
                  options = list(pageLength = 15, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("Size_MB"), digits = 2)
  })
  
  # Data summary metrics
  output$total_data_files <- renderText({
    status_data <- data_status_rv()
    if (is.null(status_data)) return("N/A")
    status_data$existing_files
  })
  
  output$total_data_size <- renderText({
    status_data <- data_status_rv()
    if (is.null(status_data)) return("N/A")
    paste0(round(status_data$total_size_mb, 1), " MB")
  })
  
  # Data directory info
  output$data_directory_info <- renderText({
    status_data <- data_status_rv()
    if (is.null(status_data)) return("Click 'Refresh Data Status' to load information")
    
    paste0(
      "Data Directory: ", status_data$data_directory, "\n",
      "Total Files: ", status_data$total_files, "\n",
      "Existing Files: ", status_data$existing_files, "\n",
      "Total Size: ", round(status_data$total_size_mb, 1), " MB\n",
      "Last Updated: ", Sys.time()
    )
  })
}

# Run the application
shinyApp(ui = ui, server = server)

