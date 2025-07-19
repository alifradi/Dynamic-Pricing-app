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
      "))
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
              column(9,
                actionButton("generate_scenario_btn", "Generate Scenario & Save CSV", 
                           class = "btn-primary btn-block", icon = icon("download"))
              )
            )
          )
        ),
        fluidRow(
          column(4,
            box(
              title = "Bandit Simulation", status = "info", solidHeader = TRUE, width = 12,
              actionButton("run_bandit_btn", "Run Bandit Simulation", 
                         class = "btn-info btn-block", icon = icon("random")),
              verbatimTextOutput("bandit_status")
            )
          ),
          column(4,
            box(
              title = "User Dynamic Price Sensitivity", status = "warning", solidHeader = TRUE, width = 12,
              actionButton("generate_dps_btn", "Generate DPS CSV", 
                         class = "btn-warning btn-block", icon = icon("chart-line")),
              verbatimTextOutput("dps_status")
            )
          ),
          column(4,
            box(
              title = "Conversion Probabilities", status = "success", solidHeader = TRUE, width = 12,
              actionButton("generate_conversion_btn", "Generate Conversion CSV", 
                         class = "btn-success btn-block", icon = icon("percent")),
              verbatimTextOutput("conversion_status")
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
    conversion_refresh = 0
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
  
  # Generate scenario and save CSV
  observeEvent(input$generate_scenario_btn, {
    tryCatch({
      res <- POST(paste0(API_URL, "/sample_offers_for_users"), 
                 query = list(
                   num_users = input$num_users,
                   num_hotels = input$num_hotels,
                   num_partners = input$num_partners,
                   days_to_go = input$days_to_go,
                   days_var = input$days_var
                 ))
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        showNotification("Scenario generated and CSV saved successfully", type = "default")
      } else {
        showNotification("Error generating scenario", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Run bandit simulation
  observeEvent(input$run_bandit_btn, {
    tryCatch({
      res <- POST(paste0(API_URL, "/run_bandit_simulation"))
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$bandit_data <- data
        rv$bandit_refresh <- rv$bandit_refresh + 1  # Trigger refresh
        showNotification("Bandit simulation completed", type = "default")
      } else {
        showNotification("Error running bandit simulation", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Generate user dynamic price sensitivity CSV
  observeEvent(input$generate_dps_btn, {
    tryCatch({
      res <- POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"))
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$dps_data <- data
        rv$dps_refresh <- rv$dps_refresh + 1  # Trigger refresh
        showNotification("DPS CSV generated successfully", type = "default")
      } else {
        showNotification("Error generating DPS CSV", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Generate conversion probabilities CSV
  observeEvent(input$generate_conversion_btn, {
    tryCatch({
      res <- POST(paste0(API_URL, "/conversion_probabilities_csv"))
      
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        rv$conversion_data <- data
        rv$conversion_refresh <- rv$conversion_refresh + 1  # Trigger refresh
        showNotification("Conversion probabilities CSV generated successfully", type = "default")
      } else {
        showNotification("Error generating conversion probabilities CSV", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Output: Scenario Data Table
  output$scenario_data_table <- DT::renderDataTable({
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
    tryCatch({
      res <- GET(paste0(API_URL, "/bandit_simulation_results"))
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        if (length(data$data) > 0) {
          df <- as.data.frame(data$data)
          DT::datatable(df, 
                        options = list(pageLength = 10, scrollX = TRUE),
                        rownames = FALSE) %>%
            DT::formatRound(columns = c("probability_of_click", "true_click_prob", "preference_score"), digits = 4)
        } else {
          data.frame(Message = "No bandit data available")
        }
      } else {
        data.frame(Message = "Error loading bandit data")
      }
    }, error = function(e) {
      data.frame(Message = paste("Error:", e$message))
    })
  })
  
  # Output: DPS Table
  output$dps_table <- DT::renderDataTable({
    tryCatch({
      # Try to get data from backend API
      res <- GET(paste0(API_URL, "/user_dynamic_price_sensitivity_data"))
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        if (length(data$data) > 0) {
          df <- as.data.frame(data$data)
          DT::datatable(df, 
                        options = list(pageLength = 10, scrollX = TRUE),
                        rownames = FALSE) %>%
            DT::formatRound(columns = intersect(c("base_price_sensitivity", "dynamic_price_sensitivity", "price_mean", "price_std", "avg_days_to_go"), names(df)), digits = 4)
        } else {
          data.frame(Message = "No DPS data available. Please click 'Generate DPS CSV' button first.")
        }
      } else {
        data.frame(Message = "Error loading DPS data from backend")
      }
    }, error = function(e) {
      data.frame(Message = paste("DPS CSV not found or error loading data:", e$message))
    })
  })
  
  # Output: Conversion Table
  output$conversion_table <- DT::renderDataTable({
    tryCatch({
      # Try to get data from backend API
      res <- GET(paste0(API_URL, "/conversion_probabilities_data"))
      if (res$status_code == 200) {
        data <- fromJSON(rawToChar(res$content))
        if (length(data$data) > 0) {
          df <- as.data.frame(data$data)
          DT::datatable(df, 
                        options = list(pageLength = 10, scrollX = TRUE),
                        rownames = FALSE) %>%
            DT::formatRound(columns = intersect(c("conversion_probability"), names(df)), digits = 4)
        } else {
          data.frame(Message = "No conversion data available. Please click 'Generate Conversion CSV' button first.")
        }
      } else {
        data.frame(Message = "Error loading conversion data from backend")
      }
    }, error = function(e) {
      data.frame(Message = paste("Conversion probabilities CSV not found or error loading data:", e$message))
    })
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
    # Load market state data
    tryCatch({
      market_data <- read.csv("data/market_state_by_location.csv")
      
      ggplot(market_data, aes(x = market_state_label)) +
        geom_bar(fill = "steelblue", alpha = 0.7) +
        labs(title = "Market Demand Distribution",
             x = "Market State",
             y = "Number of Locations") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }, error = function(e) {
      ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "Market data not available") +
        theme_void()
    })
  })
  
  # Output: Price Sensitivity Distribution Plot
  output$price_sensitivity_plot <- renderPlot({
    # Load user profiles data
    tryCatch({
      user_data <- read.csv("data/enhanced_user_profiles.csv")
      
      ggplot(user_data, aes(x = price_sensitivity)) +
        geom_histogram(bins = 20, fill = "orange", alpha = 0.7) +
        labs(title = "User Price Sensitivity Distribution",
             x = "Price Sensitivity",
             y = "Frequency") +
        theme_minimal()
    }, error = function(e) {
      ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "User data not available") +
        theme_void()
    })
  })
  
  # Output: Click Probability vs Rank Plot
  output$click_probability_plot <- renderPlot({
    # Load bandit simulation data
    tryCatch({
      bandit_data <- read.csv("data/bandit_simulation_results.csv")
      
      ggplot(bandit_data, aes(x = rank, y = probability_of_click)) +
        geom_point(alpha = 0.6, color = "purple") +
        geom_smooth(method = "loess", se = TRUE, color = "red") +
        labs(title = "Click Probability vs. Rank",
             x = "Rank",
             y = "Click Probability") +
        theme_minimal()
    }, error = function(e) {
      ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "Bandit data not available") +
        theme_void()
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)

