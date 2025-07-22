# Enhanced Hotel Ranking Simulation Shiny App
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
# Try multiple ways to get the API URL
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
  last_bandit_update = NULL,
  last_dps_update = NULL,
  last_conversion_update = NULL,
  last_market_update = NULL,
  last_user_update = NULL,
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

fetch_market_data <- function() {
  tryCatch({
    read.csv("/data/market_state_by_location.csv")
  }, error = function(e) {
    NULL
  })
}

fetch_user_data <- function() {
  tryCatch({
    read.csv("/data/enhanced_user_profiles.csv")
  }, error = function(e) {
    NULL
  })
}

# --- UI Definition ---
ui <- dashboardPage(
  dashboardHeader(title = "trivago Offer Ranking Simulator"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("CSV Generation", tabName = "csv_generation", icon = icon("file-csv")),
      menuItem("Scenario & Model Inputs", tabName = "scenario", icon = icon("database")),
      menuItem("Strategy Comparison", tabName = "strategy", icon = icon("balance-scale")),
      menuItem("Dashboard & trivago Insights", tabName = "dashboard", icon = icon("chart-line"))
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
      # --- TAB 1: CSV GENERATION ---
      tabItem(tabName = "csv_generation",
        fluidRow(
          box(
            title = "CSV Generation Parameters", status = "primary", solidHeader = TRUE, width = 12,
        fluidRow(
              column(3,
                numericInput("num_users", "Number of Users:", value = 10, min = 1, max = 100)
              ),
              column(3,
                numericInput("num_hotels", "Number of Hotels per User:", value = 10, min = 1, max = 20)
              ),
              column(3,
                numericInput("num_partners", "Number of Partners:", value = 5, min = 1, max = 10)
              ),
              column(3,
                numericInput("days_to_go", "Days to Go (mean):", value = 30, min = 1, max = 365)
          )
        ),
        fluidRow(
              column(3,
                numericInput("days_var", "Days Variance:", value = 5, min = 1, max = 30)
              ),
              column(3,
                numericInput("min_users_per_destination", "Minimum Users per Destination:", value = 1, min = 1, max = 20)
              ),
              column(6,
                # Placeholder for spacing
              )
            )
          )
            ),
            fluidRow(
          column(12,
            box(
              title = "Sample Data and Run Simulation", status = "primary", solidHeader = TRUE, width = 12,
              actionButton("run_full_simulation_btn", "Sample Data and Run Simulation", 
                         class = "btn-primary btn-block btn-lg", icon = icon("play")),
              verbatimTextOutput("simulation_status")
            )
          )
        ),
        fluidRow(
          box(
            title = "Generated CSV Data", status = "info", solidHeader = TRUE, width = 12,
            tabsetPanel(
              tabPanel("Bandit Results", DT::dataTableOutput("bandit_table")),
              tabPanel("User DPS", DT::dataTableOutput("dps_table")),
              tabPanel("Conversion Probs", DT::dataTableOutput("conversion_table"))
            )
          )
        )
      ),
      
      # --- TAB 2: SCENARIO & MODEL INPUTS ---
      tabItem(tabName = "scenario",
        fluidRow(
          box(
            title = "User Selection", status = "primary", solidHeader = TRUE, width = 12,
          fluidRow(
            column(6,
                selectInput("user_id_select", "Select User ID:", choices = NULL, width = "100%")
              ),
              column(6,
                actionButton("load_user_scenario_btn", "Load User Scenario", 
                           class = "btn-primary btn-block", icon = icon("search"))
              )
            )
          )
        ),
        fluidRow(
          box(
            title = "User Scenario Data (Raw Materials for Models)", status = "info", solidHeader = TRUE, width = 12,
            DT::dataTableOutput("scenario_data_table")
          )
        ),
        fluidRow(
          box(
            title = "Probability Evolution Analysis", status = "warning", solidHeader = TRUE, width = 12,
            fluidRow(
              column(4,
                selectInput("prob_user_select", "Select User ID for Analysis:", choices = NULL, width = "100%")
              ),
              column(4,
                selectInput("prob_offer_select", "Select Offer ID for Analysis:", choices = NULL, width = "100%")
              ),
              column(4,
                actionButton("load_prob_evolution_btn", "Load Probability Evolution", 
                           class = "btn-warning btn-block", icon = icon("chart-line"))
              )
            ),
            fluidRow(
              column(12,
                div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;",
                  # Initialize MathJax first
                  withMathJax(),
                  
                  h5("Mathematical Formulas:", style = "color: #495057; font-weight: bold; margin-bottom: 20px;"),
                  
                  div(style = "font-family: 'Times New Roman', serif; font-size: 14px; line-height: 1.8;",
                    # Equation 1: Improved wrapping
                    div(style = "margin-bottom: 20px; overflow-x: auto;",
                      helpText("(1) Click Probability (Theoretical)"),
                      "$$P(\\text{True Click}) = \\min\\left(0.95, \\max\\left(0.05, \\frac{\\text{preference_score}}{\\text{rank}}\\right)\\right)$$"
                    ),
                    
                    # Equation 2: Simplified notation
                    div(style = "margin-bottom: 20px;",
                      helpText("(2) Click Probability (Empirical)"),
                      "$$P(\\text{Click}) = \\bar{x}_{n} = \\bar{x}_{n-1} + \\frac{x_n - \\bar{x}_{n-1}}{n}$$"
                    ),
                    
                    # Equation 3: Standard logistic
                    div(style = "margin-bottom: 20px;",
                      helpText("(3) Conversion Probability"),
                      "$$P(\\text{Conversion}) = \\sigma(\\text{logit}) = \\frac{1}{1 + e^{-\\text{logit}}}$$"
                    ),
                    
                    # Equation 4: Multi-line format
                    div(style = "margin-bottom: 20px; overflow-x: auto;", 
                      helpText("(4) Logit Components"),
                      "$$\\begin{aligned}
                      \\text{logit} = & -0.5 \\times \\text{price\\_diff} \\\\
                      & + 1.5 \\times \\text{hotel\\_rating} \\\\
                      & + 0.8 \\times \\text{amenities} \\\\
                      & + 1.2 \\times \\text{brand} \\\\
                      & + 0.5 \\times \\text{loyalty} \\\\
                      & - 0.01 \\times \\text{price} \\times \\text{sensitivity}
                      \\end{aligned}$$"
                    ),
                    
                    # Explanation box with improved spacing
                    div(style = "margin-top: 25px; padding: 15px; background-color: #e9ecef; border-left: 4px solid #007bff; border-radius: 3px;",
                      h6("Theoretical Difference:", style = "color: #495057; font-weight: bold; margin-bottom: 12px;"),
                      p(HTML("&bull;"), tags$strong("P(True Click):"), "Theoretical probability based on preference score and rank position", style = "margin-bottom: 10px;"),
                      p(HTML("&bull;"), tags$strong("P(Click):"), "Empirical probability learned from user behavior data", style = "margin-bottom: 10px;"),
                      p(HTML("&bull;"), tags$strong("P(Conversion):"), "Booking probability given click, depends on offer characteristics")
                    )
                  )
                )
              )
            ),
            fluidRow(
              column(12,
                plotlyOutput("comprehensive_prob_evolution_plot")
              )
            )
          )
        )
      ),
      
      # --- TAB 2: STRATEGY COMPARISON ---
      tabItem(tabName = "strategy",
        fluidRow(
          box(
            title = "Strategy Selection", status = "primary", solidHeader = TRUE, width = 12,
                fluidRow(
              column(6,
                selectInput("strategy_select", "Select Ranking Strategy:", 
                           choices = c("Greedy", "User-First", "Stochastic LP", "RL Policy", "all"), 
                           selected = "all")
              ),
              column(6,
                actionButton("apply_strategy_btn", "Apply Strategy", 
                           class = "btn-success btn-block", icon = icon("play"))
              )
            )
          )
        ),
            fluidRow(
          column(3,
            div(class = "strategy-card greedy-card",
              h4("Greedy Strategy"),
              p("Maximize commission revenue"),
              DT::dataTableOutput("greedy_results_table")
            )
          ),
          column(3,
            div(class = "strategy-card user-first-card",
              h4("User-First Strategy"),
              p("Prioritize user satisfaction"),
              DT::dataTableOutput("user_first_results_table")
            )
          ),
          column(3,
            div(class = "strategy-card stochastic-card",
              h4("Stochastic LP Strategy"),
              p("Optimization-based ranking"),
              DT::dataTableOutput("stochastic_lp_results_table")
            )
          ),
          column(3,
            div(class = "strategy-card rl-card",
              h4("RL Policy Strategy"),
              p("Adaptive based on market conditions"),
              DT::dataTableOutput("rl_policy_results_table")
            )
          )
        )
      ),
      
      # --- TAB 3: DASHBOARD & TRIVAGO INSIGHTS ---
      tabItem(tabName = "dashboard",
        fluidRow(
          column(6,
            box(
              title = "Revenue vs. Trust Pareto Frontier", status = "success", solidHeader = TRUE, width = 12,
              plotOutput("pareto_frontier_plot")
            )
          ),
          column(6,
            box(
              title = "Market Demand Distribution", status = "info", solidHeader = TRUE, width = 12,
              plotOutput("market_demand_plot")
            )
          )
        ),
        fluidRow(
          column(6,
            box(
              title = "User Price Sensitivity Distribution", status = "warning", solidHeader = TRUE, width = 12,
              plotOutput("price_sensitivity_plot")
            )
          ),
          column(6,
            box(
              title = "Click Probability vs. Rank", status = "primary", solidHeader = TRUE, width = 12,
              plotOutput("click_probability_plot")
            )
          )
        ),
        fluidRow(
          box(
            title = "Strategy Performance Summary", status = "info", solidHeader = TRUE, width = 12,
            DT::dataTableOutput("strategy_summary_table")
          )
        )
      )
    )
  )
)

# --- Server Logic ---
server <- function(input, output, session) {
  
  # Reactive values
  rv <- reactiveValues(
    user_ids = NULL,
    scenario_data = NULL,
    ranking_results = NULL,
    market_data = NULL,
    bandit_data = NULL,
    dps_data = NULL,
    conversion_data = NULL,
    bandit_refresh = 0,
    dps_refresh = 0,
    conversion_refresh = 0,
    scenario_refresh = 0,
    simulation_message = NULL, # Added for simulation status
    prob_evolution_data = NULL, # New for probability evolution
    prob_evolution_refresh = 0 # New for probability evolution
  )
  
  # Initialize app
  observe({
    # Load user IDs on startup
    load_user_ids()
  })
  
  # Load user IDs from backend
  load_user_ids <- function() {
    tryCatch({
      res <- POST(paste0(API_URL, "/get_scenario_inputs"))
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$user_ids <- data$user_ids
        updateSelectInput(session, "user_id_select", choices = rv$user_ids)
      }
    }, error = function(e) {
      showNotification("Error loading user IDs", type = "error")
    })
  }
  
  # Load user scenario data
  observeEvent(input$load_user_scenario_btn, {
    if (is.null(input$user_id_select)) {
      showNotification("Please select a user ID", type = "warning")
      return()
    }
    
  tryCatch({
      res <- POST(paste0(API_URL, "/get_user_scenario"), 
                 body = list(user_id = input$user_id_select), 
                 encode = "json")
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$scenario_data <- data$scenario_data
        rv$market_data <- data$market_context
        rv$scenario_refresh <- rv$scenario_refresh + 1  # Trigger refresh
        
        showNotification("User scenario loaded successfully", type = "default")
        } else {
        showNotification("Error loading user scenario", type = "error")
        }
      }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Apply ranking strategy
  observeEvent(input$apply_strategy_btn, {
    if (is.null(input$user_id_select)) {
      showNotification("Please select a user ID first", type = "warning")
      return()
    }
    
    tryCatch({
      res <- POST(paste0(API_URL, "/rank"), 
                 body = list(user_id = input$user_id_select, strategy = input$strategy_select), 
                 encode = "json")
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$ranking_results <- data$results
        
        showNotification("Strategy applied successfully", type = "default")
      } else {
        showNotification("Error applying strategy", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Sample data and run full simulation (consolidated)
  observeEvent(input$run_full_simulation_btn, {
    tryCatch({
      # Step 1: Generate scenario and save CSV
      showNotification("Step 1/4: Generating scenario data...", type = "message")
      res1 <- POST(paste0(API_URL, "/sample_offers_for_users"), 
                 query = list(
                   num_users = input$num_users,
                   num_hotels = input$num_hotels,
                   num_partners = input$num_partners,
                   days_to_go = input$days_to_go,
                   days_var = input$days_var,
                   min_users_per_destination = input$min_users_per_destination
                 ))
      
      if (res1$status_code != 200) {
        error_data <- fromJSON(rawToChar(res1$content))
        if (error_data$error == "constraint_violation") {
          showNotification(paste("Constraint violation: Could not find enough destinations with at least", input$min_users_per_destination, "users. Available destinations:", paste(names(error_data$available_destinations), collapse=", ")), type = "error")
      } else {
          showNotification("Error generating scenario", type = "error")
        }
        return()
      }
      
      # Parse the response to get the new metrics
      data1 <- fromJSON(rawToChar(res1$content))
      total_offers <- data1$total_offers
      offered_rooms_by_location <- data1$offered_rooms_by_location
      demand_per_destination <- data1$demand_per_destination
      offers_per_destination <- data1$offers_per_destination
      
      # Step 2: Run bandit simulation
      showNotification("Step 2/4: Running bandit simulation...", type = "message")
      res2 <- POST(paste0(API_URL, "/run_bandit_simulation"))
      
      if (res2$status_code != 200) {
        showNotification("Error running bandit simulation", type = "error")
        return()
      }
      
      # Step 3: Generate DPS CSV
      showNotification("Step 3/4: Generating DPS CSV...", type = "message")
      res3 <- POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"), 
                 query = list(
                   num_users = input$num_users,
                   num_hotels = input$num_hotels,
                   num_partners = input$num_partners,
                   days_to_go = input$days_to_go,
                   days_var = input$days_var,
                   min_users_per_destination = input$min_users_per_destination
                 ))
      
      if (res3$status_code != 200) {
        showNotification("Error generating DPS CSV", type = "error")
        return()
      }
    
      # Step 4: Generate conversion probabilities CSV
      showNotification("Step 4/4: Generating conversion probabilities CSV...", type = "message")
      res4 <- POST(paste0(API_URL, "/conversion_probabilities_csv"), 
                 query = list(
                   num_users = input$num_users,
                   num_hotels = input$num_hotels,
                   num_partners = input$num_partners,
                   days_to_go = input$days_to_go,
                   days_var = input$days_var,
                   min_users_per_destination = input$min_users_per_destination
                 ))
      
      if (res4$status_code != 200) {
        showNotification("Error generating conversion probabilities CSV", type = "error")
        return()
      }
      
      # Success - invalidate all caches and trigger refreshes
      data_cache$bandit_data <- NULL
      data_cache$dps_data <- NULL
      data_cache$conversion_data <- NULL
      data_cache$market_data <- NULL
      data_cache$last_bandit_update <- NULL
      data_cache$last_dps_update <- NULL
      data_cache$last_conversion_update <- NULL
      data_cache$last_market_update <- NULL
      
      # Trigger all refresh counters to update all tables
      rv$scenario_refresh <- rv$scenario_refresh + 1
      rv$bandit_refresh <- rv$bandit_refresh + 1
      rv$dps_refresh <- rv$dps_refresh + 1
      rv$conversion_refresh <- rv$conversion_refresh + 1
      
      # Update simulation message with new metrics
      rv$simulation_message <- paste0(
        "✅ All simulations completed successfully!\n",
        "📊 Summary:\n",
        "• Total offers generated: ", total_offers, "\n",
        "• Offered rooms by location: ", offered_rooms_by_location, "\n",
        "• Users with offers: ", data1$users_with_offers, " out of ", data1$parameters$num_users, "\n",
        "• Hotels sampled per destination: ", data1$parameters$num_hotels, "\n",
        "• Partners sampled: ", data1$parameters$num_partners, "\n",
        "• Minimum users per destination constraint: ", input$min_users_per_destination, "\n\n",
        "🏙️ Demand per destination (users per location):\n",
        paste(sapply(names(demand_per_destination), function(dest) {
          paste0("  • ", dest, ": ", demand_per_destination[[dest]], " users")
        }), collapse = "\n"), "\n\n",
        "🏨 Offers per destination (rooms × partners):\n",
        paste(sapply(names(offers_per_destination), function(dest) {
          paste0("  • ", dest, ": ", offers_per_destination[[dest]], " offers")
        }), collapse = "\n"), "\n\n",
        "✅ Constraint validation: All selected destinations have at least ", input$min_users_per_destination, " users"
      )
      
      showNotification("✅ All simulations completed successfully!", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Auto-reload user scenario when CSV files are regenerated
  observe({
    # Watch for changes in refresh counters
    rv$bandit_refresh
    rv$dps_refresh
    rv$conversion_refresh
    
    # If we have a selected user and scenario data was previously loaded, reload it
    if (!is.null(input$user_id_select) && !is.null(rv$scenario_data)) {
    tryCatch({
        res <- POST(paste0(API_URL, "/get_user_scenario"), 
                   body = list(user_id = input$user_id_select), 
                   encode = "json")
        
        if (res$status_code == 200) {
          data <- fromJSON(rawToChar(res$content))
          rv$scenario_data <- data$scenario_data
          rv$market_data <- data$market_context
          rv$scenario_refresh <- rv$scenario_refresh + 1
      }
    }, error = function(e) {
        # Silently handle errors to avoid spam
      })
    }
  })
  
  # Update probability analysis dropdowns when bandit data is available
  observe({
    rv$bandit_refresh
    
    tryCatch({
      bandit_data <- get_cached_data("bandit_data", fetch_bandit_data)
      if (!is.null(bandit_data) && length(bandit_data) > 0) {
        df <- as.data.frame(bandit_data)
        unique_users <- unique(df$user_id)
        unique_offers <- unique(df$offer_id)
        
        updateSelectInput(session, "prob_user_select", choices = unique_users, selected = unique_users[1])
        updateSelectInput(session, "prob_offer_select", choices = unique_offers, selected = unique_offers[1])
      }
    }, error = function(e) {
      # Silently handle errors
    })
  })
  
  # Load probability evolution data
  observeEvent(input$load_prob_evolution_btn, {
    tryCatch({
      if (is.null(input$prob_user_select) || is.null(input$prob_offer_select)) {
        showNotification("Please select both user ID and offer ID", type = "warning")
        return()
      }
      
      # Get bandit simulation results
      bandit_data <- get_cached_data("bandit_data", fetch_bandit_data)
      if (is.null(bandit_data)) {
        showNotification("No bandit simulation data available. Please run simulation first.", type = "warning")
        return()
      }
      
      df <- as.data.frame(bandit_data)
      
      # Convert numeric columns properly (handle comma decimal separators)
      df$rank <- as.numeric(as.character(df$rank))
      df$probability_of_click <- as.numeric(as.character(df$probability_of_click))
      df$true_click_prob <- as.numeric(as.character(df$true_click_prob))
      df$preference_score <- as.numeric(as.character(df$preference_score))
      
      # Filter data for selected user and offer
      filtered_data <- df[df$user_id == input$prob_user_select & df$offer_id == input$prob_offer_select, ]
      
      if (nrow(filtered_data) == 0) {
        showNotification("No data found for selected user and offer combination", type = "warning")
        return()
      }
      
      # Store filtered data for plotting
      rv$prob_evolution_data <- filtered_data
      rv$prob_evolution_refresh <- rv$prob_evolution_refresh + 1
      
      showNotification("Probability evolution data loaded successfully", type = "message")
      
    }, error = function(e) {
      showNotification(paste("Error loading probability evolution:", e$message), type = "error")
    })
  })
  
  # Output: Scenario Data Table
  output$scenario_data_table <- DT::renderDataTable({
    # Make reactive to refresh triggers
    rv$scenario_refresh
    rv$bandit_refresh
    rv$dps_refresh
    rv$conversion_refresh
    
    tryCatch({
      if (is.null(rv$scenario_data)) {
        return(data.frame(Message = "No scenario data loaded. Please generate CSV files first in the CSV Generation tab, then load a user scenario."))
      }
      
      if (length(rv$scenario_data) == 0) {
        return(data.frame(Message = "No scenario data available for the selected user"))
      }
      
      df <- as.data.frame(rv$scenario_data)
      
      if (nrow(df) == 0) {
        return(data.frame(Message = "Empty scenario data for the selected user"))
      }
      
      DT::datatable(df, 
                    options = list(pageLength = 10, scrollX = TRUE),
                    rownames = FALSE) %>%
        DT::formatRound(columns = intersect(c("avg_price", "probability_of_click", "conversion_probability"), names(df)), digits = 3)
    }, error = function(e) {
      data.frame(Message = paste("Error loading scenario data:", e$message))
    })
  })
  
  # Output: Greedy Results Table
  output$greedy_results_table <- DT::renderDataTable({
    if (is.null(rv$ranking_results) || is.null(rv$ranking_results$Greedy)) {
      return(data.frame(Message = "No results"))
    }
    
    df <- as.data.frame(rv$ranking_results$Greedy$ranked_list)
    
    DT::datatable(df, 
                  options = list(pageLength = 5, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("price_per_night", "greedy_score"), digits = 2)
  })
  
  # Output: User-First Results Table
  output$user_first_results_table <- DT::renderDataTable({
    if (is.null(rv$ranking_results) || is.null(rv$ranking_results$`User-First`)) {
      return(data.frame(Message = "No results"))
    }
    
    df <- as.data.frame(rv$ranking_results$`User-First`$ranked_list)
    
    DT::datatable(df, 
                  options = list(pageLength = 5, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("price_per_night", "user_first_score"), digits = 2)
  })
  
  # Output: Stochastic LP Results Table
  output$stochastic_lp_results_table <- DT::renderDataTable({
    if (is.null(rv$ranking_results) || is.null(rv$ranking_results$`Stochastic LP`)) {
      return(data.frame(Message = "No results"))
    }
    
    df <- as.data.frame(rv$ranking_results$`Stochastic LP`$ranked_list)
    
    DT::datatable(df, 
                  options = list(pageLength = 5, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("price_per_night", "stochastic_score"), digits = 2)
  })
  
  # Output: RL Policy Results Table
  output$rl_policy_results_table <- DT::renderDataTable({
    if (is.null(rv$ranking_results) || is.null(rv$ranking_results$`RL Policy`)) {
      return(data.frame(Message = "No results"))
    }
    
    df <- as.data.frame(rv$ranking_results$`RL Policy`$ranked_list)
    
    DT::datatable(df, 
                  options = list(pageLength = 5, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("price_per_night", "rl_score"), digits = 2)
  })
  
  # Output: Bandit Status
  output$bandit_status <- renderText({
    if (is.null(rv$bandit_data)) {
      return("No bandit simulation run yet")
    }
    paste("Status:", rv$bandit_data$message)
  })
  
  # Output: DPS Status
  output$dps_status <- renderText({
    if (is.null(rv$dps_data)) {
      return("No DPS CSV generated yet")
    }
    paste("Status:", rv$dps_data$message)
  })
  
  # Output: Conversion Status
  output$conversion_status <- renderText({
    if (is.null(rv$conversion_data)) {
      return("No conversion CSV generated yet")
    }
    paste("Status:", rv$conversion_data$message)
  })
  
  # Output: Bandit Table
  output$bandit_table <- DT::renderDataTable({
    # Use cached data instead of API call
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
  
  # Output: DPS Table
  output$dps_table <- DT::renderDataTable({
    # Use cached data instead of API call
    dps_data <- get_cached_data("dps_data", fetch_dps_data)
    
    if (is.null(dps_data) || length(dps_data) == 0) {
      return(data.frame(Message = "No DPS data available. Please click 'Generate DPS CSV' button first."))
    }
    
    df <- as.data.frame(dps_data)
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = intersect(c("base_price_sensitivity", "dynamic_price_sensitivity", "price_mean", "price_std", "avg_days_to_go"), names(df)), digits = 4)
  })
  
  # Output: Conversion Table
  output$conversion_table <- DT::renderDataTable({
    # Use cached data instead of API call
    conversion_data <- get_cached_data("conversion_data", fetch_conversion_data)
    
    if (is.null(conversion_data) || length(conversion_data) == 0) {
      return(data.frame(Message = "No conversion data available. Please click 'Generate Conversion CSV' button first."))
    }
    
    df <- as.data.frame(conversion_data)
    DT::datatable(df, 
                  options = list(pageLength = 10, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = intersect(c("conversion_probability"), names(df)), digits = 4)
  })
  
  # Output: Strategy Summary Table
  output$strategy_summary_table <- DT::renderDataTable({
    if (is.null(rv$ranking_results)) {
      return(data.frame(Message = "No ranking results available"))
    }
    
    # Create summary table
    summary_data <- data.frame(
      Strategy = names(rv$ranking_results),
      Total_Revenue = sapply(rv$ranking_results, function(x) sum(x$ranked_list$price_per_night, na.rm = TRUE)),
      Avg_Trust_Score = sapply(rv$ranking_results, function(x) mean(x$ranked_list$trust_score, na.rm = TRUE)),
      Conversion_Rate = sapply(rv$ranking_results, function(x) mean(x$ranked_list$conversion_probability, na.rm = TRUE))
    )
    
    DT::datatable(summary_data, 
                  options = list(pageLength = 5, scrollX = TRUE),
                  rownames = FALSE) %>%
      DT::formatRound(columns = c("Total_Revenue", "Avg_Trust_Score", "Conversion_Rate"), digits = 2)
  })
  
  # Output: Pareto Frontier Plot
  output$pareto_frontier_plot <- renderPlot({
    if (is.null(rv$ranking_results)) {
      return(ggplot() + 
               annotate("text", x = 0.5, y = 0.5, label = "No ranking results available") +
               theme_void())
    }
    
    # Create Pareto frontier data
    pareto_data <- data.frame(
      Strategy = names(rv$ranking_results),
      Revenue = sapply(rv$ranking_results, function(x) sum(x$ranked_list$price_per_night, na.rm = TRUE)),
      Trust = sapply(rv$ranking_results, function(x) mean(x$ranked_list$trust_score, na.rm = TRUE))
    )
    
    ggplot(pareto_data, aes(x = Revenue, y = Trust, label = Strategy)) +
      geom_point(size = 3, color = "steelblue") +
      geom_text(vjust = -0.5, hjust = 0.5) +
      labs(title = "Revenue vs. Trust Pareto Frontier",
           x = "Total Revenue",
           y = "Average Trust Score") +
      theme_minimal()
  })
  
  # Output: Market Demand Distribution Plot
  output$market_demand_plot <- renderPlot({
    # Use cached market data
    market_data <- get_cached_data("market_data", fetch_market_data)
    
    if (is.null(market_data)) {
      return(ggplot() + 
               annotate("text", x = 0.5, y = 0.5, label = "Market data not available") +
               theme_void())
    }
    
    ggplot(market_data, aes(x = market_state_label)) +
      geom_bar(fill = "steelblue", alpha = 0.7) +
      labs(title = "Market Demand Distribution",
           x = "Market State",
           y = "Number of Locations") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # Output: Price Sensitivity Distribution Plot
  output$price_sensitivity_plot <- renderPlot({
    # Use cached user data
    user_data <- get_cached_data("user_data", fetch_user_data)
    
    if (is.null(user_data)) {
      return(ggplot() + 
               annotate("text", x = 0.5, y = 0.5, label = "User data not available") +
               theme_void())
    }
    
    ggplot(user_data, aes(x = price_sensitivity)) +
      geom_histogram(bins = 20, fill = "orange", alpha = 0.7) +
      labs(title = "User Price Sensitivity Distribution",
           x = "Price Sensitivity",
           y = "Frequency") +
      theme_minimal()
  })
  
  # Output: Click Probability vs Rank Plot
  output$click_probability_plot <- renderPlot({
    # Use cached bandit data
    bandit_data <- get_cached_data("bandit_data", fetch_bandit_data)
    
    if (is.null(bandit_data) || length(bandit_data) == 0) {
      return(ggplot() + 
               annotate("text", x = 0.5, y = 0.5, label = "Bandit data not available") +
               theme_void())
    }
    
    df <- as.data.frame(bandit_data)
    ggplot(df, aes(x = rank, y = probability_of_click)) +
      geom_point(alpha = 0.6, color = "purple") +
      geom_smooth(method = "loess", se = TRUE, color = "red") +
      labs(title = "Click Probability vs. Rank",
           x = "Rank",
           y = "Click Probability") +
      theme_minimal()
  })

  # Output: Comprehensive Probability Evolution Plot
  output$comprehensive_prob_evolution_plot <- renderPlotly({
    rv$prob_evolution_refresh
    
    tryCatch({
      if (is.null(rv$prob_evolution_data)) {
        return(plot_ly() %>% 
                 add_annotations(text = "No data available. Please load probability evolution data.", 
                               showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
      }
      
      data <- rv$prob_evolution_data
      
      # Debug: Print data structure
      print(paste("Data rows:", nrow(data)))
      print(paste("Data columns:", paste(names(data), collapse = ", ")))
      print(paste("Data types:", paste(sapply(data, class), collapse = ", ")))
      
      # Ensure rank is numeric and handle decimal conversion
      data$rank <- as.numeric(as.character(data$rank))
      data$probability_of_click <- as.numeric(as.character(data$probability_of_click))
      data$true_click_prob <- as.numeric(as.character(data$true_click_prob))
      
      # Remove any NA values
      data <- data[!is.na(data$rank) & !is.na(data$probability_of_click) & !is.na(data$true_click_prob), ]
      
      if (nrow(data) == 0) {
        return(plot_ly() %>% 
                 add_annotations(text = "No valid data found after conversion", 
                               showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5))
      }
      
      # Get conversion probability data
      conversion_data <- get_cached_data("conversion_data", fetch_conversion_data)
      conv_prob <- NULL
      if (!is.null(conversion_data)) {
        conv_df <- as.data.frame(conversion_data)
        user_conv <- conv_df[conv_df$user_id == input$prob_user_select & conv_df$offer_id == input$prob_offer_select, ]
        
        if (nrow(user_conv) > 0) {
          conv_prob <- user_conv$conversion_probability[1]
        }
      }
      
      # Create comprehensive plot showing all probabilities
      p <- plot_ly() %>%
        add_trace(data = data, x = ~rank, y = ~probability_of_click, 
                 type = 'scatter', mode = 'lines+markers', name = 'Learned P(Click)', 
                 line = list(color = '#1f77b4', width = 3),
                 marker = list(size = 8, color = '#1f77b4')) %>%
        add_trace(data = data, x = ~rank, y = ~true_click_prob, 
                 type = 'scatter', mode = 'lines+markers', name = 'True P(Click)', 
                 line = list(color = '#ff7f0e', width = 3, dash = 'dash'),
                 marker = list(size = 8, color = '#ff7f0e', symbol = 'diamond'))
      
      # Add conversion probability as horizontal line if available
      if (!is.null(conv_prob)) {
        p <- p %>% add_trace(
          x = c(min(data$rank), max(data$rank)), 
          y = c(conv_prob, conv_prob),
          type = 'scatter', 
          mode = 'lines', 
          name = paste('P(Conversion) =', round(conv_prob, 4)),
          line = list(color = '#d62728', width = 3, dash = 'solid')
        )
      }
      
      p <- p %>% layout(
        title = list(
          text = "Comprehensive Probability Analysis by Rank<br><sub>Shows Click Probabilities (Learned vs True) and Conversion Probability</sub>",
          font = list(size = 14)
        ),
        xaxis = list(title = "Rank Position", gridcolor = '#f0f0f0', type = 'linear'),
        yaxis = list(title = "Probability", range = c(0, 1), gridcolor = '#f0f0f0'),
        hovermode = 'x unified',
        legend = list(orientation = 'h', x = 0.5, y = -0.2),
        margin = list(l = 50, r = 50, t = 80, b = 80),
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
      )
      
      # Add user/offer information to hover text
      p <- p %>% add_annotations(
        text = paste("User:", input$prob_user_select, "| Offer:", input$prob_offer_select),
        showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 1.02,
        font = list(size = 12, color = '#666666')
      )
      
      p
      
    }, error = function(e) {
      print(paste("Plot error:", e$message))
      plot_ly() %>% 
        add_annotations(text = paste("Error:", e$message), 
                       showarrow = FALSE, xref = "paper", yref = "paper", x = 0.5, y = 0.5)
    })
  })

  # Simulation status output
  output$simulation_status <- renderText({
    if (is.null(rv$simulation_message)) {
      return("Click 'Sample Data and Run Simulation' to start...")
    }
    rv$simulation_message
  })
}

# Run the application
shinyApp(ui = ui, server = server)

