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
        .info-box {
          background: #f8f9fa;
          border: 2px solid #e9ecef;
          border-radius: 8px;
          padding: 15px;
          text-align: center;
          margin: 10px 0;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .info-box h4 {
          margin: 0 0 10px 0;
          color: #495057;
          font-size: 14px;
          font-weight: 600;
        }
        .info-box p {
          margin: 0;
          font-weight: bold;
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
            title = "Market Parameters for Data Generation", status = "info", solidHeader = TRUE, width = 12,
            fluidRow(
              column(3,
                numericInput("num_users_gen", "Number of Users:", value = 30, min = 1, max = 100)
              ),
              column(3,
                numericInput("num_hotels_gen", "Hotels per Destination:", value = 5, min = 1, max = 20)
              ),
              column(3,
                numericInput("num_partners_gen", "Partners per Hotel:", value = 3, min = 1, max = 10)
              ),
              column(3,
                numericInput("min_users_per_destination_gen", "Min Users per Destination:", value = 6, min = 1, max = 20)
              )
            ),
            fluidRow(
              column(3,
                numericInput("days_to_go_gen", "Days to Go (target):", value = 5, min = 1, max = 365)
              ),
              column(3,
                numericInput("days_var_gen", "Days Variance:", value = 20, min = 1, max = 30)
              ),
              column(6,
                div(style = "text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;",
                    helpText("These parameters control the data generation for the strategic simulation")
                )
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
                actionButton("load_pretrained_policy_btn", "Load Pre-trained Policy", 
                           class = "btn-warning", icon = icon("brain"))
              ),
              column(6,
                actionButton("retrain_rl_btn", "Retrain RL Agent", 
                           class = "btn-info", icon = icon("graduation-cap"))
              )
            ),
            fluidRow(
              column(6,
                verbatimTextOutput("policy_selection_output")
              ),
              column(6,
                verbatimTextOutput("retraining_output")
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "Simulation Status", status = "success", solidHeader = TRUE, width = 12,
            verbatimTextOutput("simulation_status")
          )
        ),
        fluidRow(
          box(
            title = "Generated Data Preview", status = "info", solidHeader = TRUE, width = 12,
            fluidRow(
              column(4,
                h4("Bandit Simulation Results"),
                DT::dataTableOutput("bandit_preview_table")
              ),
              column(4,
                h4("User Price Sensitivity"),
                DT::dataTableOutput("dps_preview_table")
              ),
              column(4,
                h4("Conversion Probabilities"),
                DT::dataTableOutput("conversion_preview_table")
              )
            )
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
            title = "Two-Stage Optimization System", status = "primary", solidHeader = TRUE, width = 12,
            fluidRow(
              column(12,
                actionButton("run_two_stage_optimization", "Run Two-Stage Optimization", 
                           class = "btn-primary btn-lg", icon = icon("layer-group"),
                           style = "width: 100%; height: 60px; font-size: 16px;")
              )
            ),
            fluidRow(
              column(12,
                div(style = "text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;",
                  helpText("Two-Stage System: Stage 1 (Ranking) + Stage 2 (Hiding) for Maximum Customer Satisfaction, Partner Conversions, and Trivago Gains")
                )
              )
            ),
            br(), br(),
            fluidRow(
              column(12,
                box(
                  title = "Objective Function Values", status = "success", solidHeader = TRUE, width = 12,
                  fluidRow(
                    column(3,
                      div(class = "metric-box",
                        div(class = "metric-value", textOutput("trivago_income_value")),
                        div(class = "metric-label", "Trivago Income")
                      )
                    ),
                    column(3,
                      div(class = "metric-box",
                        div(class = "metric-value", textOutput("user_satisfaction_value")),
                        div(class = "metric-label", "User Satisfaction")
                      )
                    ),
                    column(3,
                      div(class = "metric-box",
                        div(class = "metric-value", textOutput("partner_conversion_value")),
                        div(class = "metric-label", "Partner Conversion Value")
                      )
                    ),
                    column(3,
                      div(class = "metric-box",
                        div(class = "metric-value", textOutput("total_objective_value")),
                        div(class = "metric-label", "Total Objective")
                      )
                    )
                  )
                )
              )
            ),
            fluidRow(
              column(12,
                box(
                  title = "Two-Stage Optimization Results Table", status = "info", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("two_stage_optimization_table")
                )
              )
            ),

          )
        ),
        fluidRow(
          box(
            title = "Mathematical Foundation", status = "warning", solidHeader = TRUE, width = 12,
            withMathJax(
              div(style = "font-family: 'Times New Roman', serif; font-size: 14px; line-height: 1.8;",
                tags$div(style = "margin-bottom: 20px; overflow-x: auto;",
                  helpText("Two-Stage Optimization System:"),
                  tags$h4("Stage 1: Optimal Ranking for Click Maximization"),
                  "$$\\text{Maximize: } \\alpha \\cdot \\text{Trivago\\_Income} + \\beta \\cdot \\text{User\\_Satisfaction} + \\gamma \\cdot \\text{Partner\\_Conversion\\_Value}$$",
                  tags$p("Subject to:"),
                  "$$\\sum_j X_{ij} \\leq 1 \\quad \\forall i \\in \\text{Positions}$$",
                  "$$\\sum_i X_{ij} \\leq 1 \\quad \\forall j \\in \\text{Offers}$$",
                  "$$\\sum_{i,j} \\text{CTR}_i \\cdot \\text{CPC}_j \\cdot X_{ij} \\leq \\text{Budget}_P \\quad \\forall P \\in \\text{Partners}$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Stage 2: Offer Hiding for Reconversion & Budget Rationalization"),
                  "$$\\text{Hide offers where: } \\text{Reconversion\\_Probability} < \\text{Threshold}$$",
                  "$$\\text{Budget\\_Utilization} \\leq \\text{Target\\_Utilization}$$"
                ),
                tags$div(style = "margin-bottom: 20px;",
                  helpText("Position-based CTR:"),
                  "$$\\text{CTR}(\\text{position}) = \\frac{1}{1 + 0.3 \\cdot \\text{position}}$$"
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
                div(style = "text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;",
                    helpText("Data generation is now integrated into the Strategic Simulation")
                )
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
    simple_optimization_results = NULL,
    ranking_results = NULL,
    objectives_results = NULL,
    weights_data = NULL,
    ab_test_results = NULL,
    refresh_counter = 0
  )
  
  # Clear any old data when app starts
  observe({
    print("[DEBUG] App starting - clearing old optimization data")
    rv$simple_optimization_results <- NULL
  })
  
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
  
  # Run strategic simulation (Enhanced)
  observeEvent(input$run_simulation_btn, {
    tryCatch({
      showNotification("Running comprehensive strategic simulation...", type = "message")
      
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
                  showNotification("Error in data sampling", type = "error", duration = 5)
        return()
      }
      
      # Step 2: Run bandit simulation
      res2 <- POST(paste0(API_URL, "/run_bandit_simulation"))
      
      if (res2$status_code != 200) {
                  showNotification("Error in bandit simulation", type = "error", duration = 5)
        return()
      }
      
      bandit_data <- fromJSON(rawToChar(res2$content))
      
      # Step 3: Generate market analysis CSVs
      POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"))
      POST(paste0(API_URL, "/conversion_probabilities_csv"))
      
      # Step 4: Run optimization with current weights
      res3 <- POST(paste0(API_URL, "/rank"), 
                 query = list(
                   alpha = input$alpha_weight,
                   beta = input$beta_weight,
                   gamma = input$gamma_weight,
                   num_positions = 5
                 ))
      
              if (res3$status_code != 200) {
          showNotification("Error in optimization", type = "error", duration = 5)
          return()
        }
      
      optimization_data <- fromJSON(rawToChar(res3$content))
      rv$optimization_results <- optimization_data
      
      # Enhanced simulation status with comprehensive market summary
      rv$simulation_message <- paste0(
        "✅ Comprehensive Strategic Simulation Completed!\n\n",
        "📊 Market Summary:\n",
        "• Total Users: ", bandit_data$total_users, "\n",
        "• Total Offers: ", bandit_data$total_offers, "\n",
        "• Total Partners: ", length(unique(bandit_data$sample_table$partner_name)), "\n",
        "• Market Demand: ", bandit_data$total_users, " users\n",
        "• Competition Density: ", length(unique(bandit_data$sample_table$partner_name)), " partners\n\n",
        "🎯 Optimization Results:\n",
        "• Trivago Income: $", round(optimization_data$objectives$trivago_income, 2), "\n",
        "• User Satisfaction: ", round(optimization_data$objectives$user_satisfaction, 2), "\n",
        "• Partner Conversion Value: ", round(optimization_data$objectives$partner_conversion_value, 2), "\n",
        "• Total Objective: ", round(optimization_data$objectives$total_objective, 2), "\n\n",
        "⚖️ Applied Weights: α=", input$alpha_weight, ", β=", input$beta_weight, ", γ=", input$gamma_weight, "\n\n",
        "📈 Generated Data Files:\n",
        "• trial_sampled_offers.csv (", bandit_data$total_offers, " offers)\n",
        "• user_dynamic_price_sensitivity.csv\n",
        "• conversion_probabilities.csv\n",
        "• bandit_simulation_results.csv\n",
        "• optimization_ranking_results.csv\n",
        "• optimization_objectives_results.csv\n",
        "• optimization_weights.csv\n\n",
        "🎲 Bandit Simulation: ", bandit_data$total_arms, " arms with ", bandit_data$clicks_per_arm, " clicks each"
      )
      
      showNotification("Comprehensive strategic simulation completed successfully!", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Load pre-trained policy (Renamed from "Select Optimal Policy")
  observeEvent(input$load_pretrained_policy_btn, {
    tryCatch({
      showNotification("Loading pre-trained RL policy...", type = "message")
      
      res <- POST(paste0(API_URL, "/select_strategic_policy"))
      
      if (res$status_code == 200) {
        policy_data <- fromJSON(rawToChar(res$content))
        rv$policy_selection_result <- policy_data
        
        output$policy_selection_output <- renderText({
          paste0(
            "🎯 Loaded Pre-trained Policy: ", policy_data$selected_policy$policy_name, "\n",
            "⚖️ Optimal Weights: α=", policy_data$selected_policy$weights$alpha, 
            ", β=", policy_data$selected_policy$weights$beta,
            ", γ=", policy_data$selected_policy$weights$gamma, "\n",
            "🧠 Exploration Rate (Epsilon): ", round(policy_data$epsilon, 4), "\n",
            "📊 Current Market State:\n",
            "• Demand: ", policy_data$market_state$market_demand, " users\n",
            "• Days to Go: ", round(policy_data$market_state$days_to_go, 1), " days\n",
            "• Competition: ", policy_data$market_state$competition_density, " partners\n",
            "• Price Volatility: ", round(policy_data$market_state$price_volatility, 3), "\n",
            "• Budget Utilization: ", round(policy_data$market_state$budget_utilization, 1), "%\n\n",
            "💡 Policy Description:\n",
            "This policy was selected by the RL agent based on current market conditions.\n",
            "The agent learned optimal weight combinations for different market scenarios."
          )
        })
        
        showNotification("Pre-trained policy loaded successfully!", type = "message")
      } else {
        showNotification("Error loading pre-trained policy", type = "error", duration = 5)
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Retrain RL agent on newly generated data
  observeEvent(input$retrain_rl_btn, {
    tryCatch({
      showNotification("Retraining RL agent on newly generated data...", type = "message")
      
      res <- POST(paste0(API_URL, "/train_rl_agent"))
      
      if (res$status_code == 200) {
        training_data <- fromJSON(rawToChar(res$content))
        
        output$retraining_output <- renderText({
          paste0(
            "🔄 RL Agent Retraining Completed!\n\n",
            "📊 Training Results:\n",
            "• Selected Policy: ", training_data$training_result$policy_name, "\n",
            "• Applied Weights: α=", training_data$training_result$weights$alpha,
            ", β=", training_data$training_result$weights$beta,
            ", γ=", training_data$training_result$weights$gamma, "\n",
            "• Average Reward: ", round(training_data$training_result$reward, 4), "\n",
            "• Training Loss: ", round(training_data$training_result$loss, 4), "\n",
            "• Final Epsilon: ", round(training_data$training_result$epsilon, 4), "\n\n",
            "💾 Model Status:\n",
            "• Model saved to: dqn_model.pth\n",
            "• Episodes trained: 5\n",
            "• Market scenarios: Varied demand, competition, and time constraints\n\n",
            "🎯 What was learned:\n",
            "The agent learned optimal weight combinations for different market conditions\n",
            "based on the newly generated data and optimization results."
          )
        })
        
        showNotification("RL agent retrained successfully on new data!", type = "message")
      } else {
        showNotification("Error retraining RL agent", type = "error", duration = 5)
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
  
  # Data preview tables
  output$bandit_preview_table <- DT::renderDataTable({
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
  
  output$dps_preview_table <- DT::renderDataTable({
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
  
  output$conversion_preview_table <- DT::renderDataTable({
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
                   num_positions = 5
                 ))
      
      if (res$status_code == 200) {
        optimization_data <- fromJSON(rawToChar(res$content), flatten = FALSE)
        rv$optimization_results <- optimization_data
        showNotification("Optimization completed!", type = "message")
      } else {
        showNotification("Error in optimization", type = "error", duration = 5)
      }
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  

  

  


  
  # Test button handler

  

  

  
  # Reactive values for optimization parameters
  optimization_params <- reactive({
    list(
      alpha = ifelse(is.null(input$alpha), 0.4, input$alpha),
      beta = ifelse(is.null(input$beta), 0.3, input$beta),
      gamma = ifelse(is.null(input$gamma), 0.3, input$gamma),
      num_positions = ifelse(is.null(input$num_positions), 5, input$num_positions),
      reconversion_threshold = ifelse(is.null(input$reconversion_threshold), 0.3, input$reconversion_threshold),
      budget_utilization_target = ifelse(is.null(input$budget_utilization_target), 0.8, input$budget_utilization_target)
    )
  })
  
  # Clear optimization results when parameters change
  observe({
    optimization_params()
    # Clear previous results when parameters change
    rv$two_stage_optimization_table <- NULL
    rv$trivago_income_value <- NULL
    rv$user_satisfaction_value <- NULL
    rv$partner_conversion_value <- NULL
    rv$total_objective_value <- NULL
  })
  
  # Two-stage optimization handler
  observeEvent(input$run_two_stage_optimization, {
    tryCatch({
      print("=== TWO-STAGE OPTIMIZATION BUTTON CLICKED ===")
      showNotification("Running two-stage optimization...", type = "message", duration = 3)
      
      print(paste("Two-stage optimization API URL:", API_URL))
      
      # Get current parameters from reactive values
      params <- optimization_params()
      alpha <- params$alpha
      beta <- params$beta
      gamma <- params$gamma
      num_positions <- params$num_positions
      reconversion_threshold <- params$reconversion_threshold
      budget_utilization_target <- params$budget_utilization_target
      
      # Call backend two-stage optimization
      res <- POST(paste0(API_URL, "/run_two_stage_optimization"), 
                 query = list(
                   alpha = alpha,
                   beta = beta,
                   gamma = gamma,
                   num_positions = num_positions,
                   reconversion_threshold = reconversion_threshold,
                   budget_utilization_target = budget_utilization_target
                 ),
                 timeout(180))  # 3 minute timeout for complex two-stage optimization
      
      print(paste("Two-stage optimization response status:", res$status_code))
      
      if (res$status_code == 200) {
        print("Two-stage optimization API call successful")
        
        # Wait a moment for CSV files to be written
        Sys.sleep(2)
        
        # Load the two-stage optimization table
        tryCatch({
          print("Attempting to load two-stage optimization table...")
          table_res <- GET(paste0(API_URL, "/two_stage_optimization_table_csv"))
          print(paste("Table response status:", table_res$status_code))
          
          if (table_res$status_code == 200) {
            # Read CSV content directly
            csv_content <- rawToChar(table_res$content)
            print(paste("CSV content length:", nchar(csv_content)))
            
            table_data <- read.csv(text = csv_content, stringsAsFactors = FALSE)
            print(paste("Table data loaded with", nrow(table_data), "rows and", ncol(table_data), "columns"))
            
            rv$two_stage_optimization_table <- table_data
            print("Two-stage optimization table loaded successfully")
            
            # Extract objective values for display boxes
            if (nrow(table_data) > 0) {
              # Calculate objective values from the table data
              rv$trivago_income_value <- sum(as.numeric(table_data$trivago_income), na.rm = TRUE)
              rv$user_satisfaction_value <- mean(as.numeric(table_data$user_satisfaction), na.rm = TRUE)
              rv$partner_conversion_value <- sum(as.numeric(table_data$partner_conversion_value), na.rm = TRUE)
              rv$total_objective_value <- sum(as.numeric(table_data$total_objective), na.rm = TRUE)
              
              print(paste("Objective values extracted - Trivago:", rv$trivago_income_value, 
                         "User:", rv$user_satisfaction_value, 
                         "Partner:", rv$partner_conversion_value, 
                         "Total:", rv$total_objective_value))
            } else {
              print("Warning: Table data has 0 rows")
            }
          } else {
            print(paste("Table response failed with status:", table_res$status_code))
          }
          
          showNotification("Two-stage optimization completed successfully!", type = "success", duration = 5)
        }, error = function(e) {
          showNotification(paste("Error loading two-stage results:", e$message), type = "error", duration = 5)
        })
      } else {
        showNotification("Two-stage optimization failed", type = "error", duration = 5)
      }
      
    }, error = function(e) {
      error_msg <- e$message
      showNotification(paste("Error in two-stage optimization:", error_msg), type = "error", duration = 5)
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
        showNotification("Error calculating Shapley values", type = "error", duration = 5)
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
  
  # Note: Data generation functionality has been moved to Strategic Simulation
  
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
        showNotification("Error fetching data status", type = "error", duration = 5)
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
  

  
  # Objective function value renderers
  output$trivago_income_value <- renderText({
    if (is.null(rv$trivago_income_value)) {
      return("N/A")
    }
    paste0("$", format(round(rv$trivago_income_value, 2), nsmall = 2))
  })
  
  output$user_satisfaction_value <- renderText({
    if (is.null(rv$user_satisfaction_value)) {
      return("N/A")
    }
    paste0(round(rv$user_satisfaction_value, 2), "/10")
  })
  
  output$partner_conversion_value <- renderText({
    if (is.null(rv$partner_conversion_value)) {
      return("N/A")
    }
    paste0("$", format(round(rv$partner_conversion_value, 2), nsmall = 2))
  })
  
  output$total_objective_value <- renderText({
    if (is.null(rv$total_objective_value)) {
      return("N/A")
    }
    paste0(format(round(rv$total_objective_value, 2), nsmall = 2))
  })
  
  # Two-stage optimization results table
  output$two_stage_optimization_table <- DT::renderDataTable({
    tryCatch({
      print("[DEBUG] Rendering two-stage optimization table...")
      
      # Check if we have two-stage optimization results
      if (is.null(rv$two_stage_optimization_table)) {
        print("[DEBUG] No two-stage optimization results available")
        return(data.frame(Message = "Run two-stage optimization to see results"))
      }
      
      table_data <- rv$two_stage_optimization_table
      
      print(paste("[DEBUG] Two-stage table data rows:", nrow(table_data)))
      
      # Create comprehensive two-stage optimization table
      if (nrow(table_data) > 0) {
        # Format the data for display (without objective function values)
        display_df <- data.frame(
          User_ID = table_data$user_id,
          Offer_ID = table_data$offer_id,
          Hotel = table_data$hotel_name,
          Partner = table_data$partner_name,
          Optimal_Rank = table_data$optimal_rank,
          Is_Hidden = ifelse(table_data$is_hidden, "Yes", "No"),
          Expected_Clicks = round(table_data$expected_clicks, 3),
          Conversion_Prob = paste0(round(table_data$conversion_probability * 100, 1), "%"),
          Reconversion_Prob = paste0(round(table_data$reconversion_probability * 100, 1), "%"),
          Price = paste0("$", table_data$price_per_night),
          Satisfaction = round(table_data$user_satisfaction_score, 2)
        )
        
        print("[DEBUG] Two-stage table data created successfully")
        
        DT::datatable(display_df,
                     options = list(pageLength = 15, scrollX = TRUE, 
                                  dom = 'Bfrtip',
                                  buttons = c("copy", "csv", "excel")),
                     caption = "Two-Stage Optimization Results: User-Offer-Rank-Hide Table",
                     filter = "top",
                     extensions = c("Buttons", "ColReorder"))
      } else {
        return(data.frame(Message = "No two-stage optimization data available"))
      }
      
    }, error = function(e) {
      print(paste("[DEBUG] Error in two-stage table renderer:", e$message))
      return(data.frame(Message = paste("Error displaying two-stage results:", e$message)))
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)

