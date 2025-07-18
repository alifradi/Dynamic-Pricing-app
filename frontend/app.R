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
  dashboardHeader(title = "Enhanced Hotel Ranking Simulation"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Scenario Setup", tabName = "scenario", icon = icon("cogs")),
      menuItem("Strategy Selection", tabName = "strategy", icon = icon("chart-line")),
      menuItem("Results & Analysis", tabName = "results", icon = icon("chart-bar")),
      menuItem("Market Analysis", tabName = "market", icon = icon("globe")),
      menuItem("Generated Data", tabName = "generated_data", icon = icon("table"))
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
        .user-profile-box {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 8px;
          padding: 15px;
          margin: 10px 0;
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
        .recommendation-box {
          background: #e8f5e8;
          border-left: 4px solid #28a745;
          padding: 10px;
          margin: 10px 0;
          border-radius: 4px;
        }
        .warning-box {
          background: #fff3cd;
          border-left: 4px solid #ffc107;
          padding: 10px;
          margin: 10px 0;
          border-radius: 4px;
        }
      "))
    ),
    
    # Test button for debugging
    div(style = "position: fixed; top: 10px; right: 10px; z-index: 1000;",
        actionButton("test_btn", "Test App", class = "btn-warning btn-sm")
    ),
    
    tabItems(
      # --- TAB 1: SCENARIO SETUP ---
      tabItem(tabName = "scenario",
        fluidRow(
          # User Profile Display only
          box(
            title = "Selected User Profile", status = "info", solidHeader = TRUE, width = 12,
            uiOutput("user_profile_display")
          )
        ),
        fluidRow(
          box(
            title = "Sampled Offers for Selected User", status = "success", solidHeader = TRUE, width = 12,
            div(style = "margin-bottom: 15px;",
                selectInput("user_profile_select_offers", "Choose User ID:", choices = NULL, width = "100%")
            ),
            fluidRow(
              column(3, numericInput("num_hotels", "Number of Hotels:", value = 10, min = 1, max = 20)),
              column(3, numericInput("num_partners", "Number of Partners:", value = 5, min = 1, max = 10)),
              column(3, sliderInput("days_to_go", "Days to Go (Check-in):", min = 1, max = 180, value = 30)),
              column(3, sliderInput("days_var", "Days Variance:", min = 1, max = 30, value = 5))
            ),
            fluidRow(
              column(3, numericInput("num_users", "Number of Users to Sample:", value = 1, min = 1, max = 20))
            ),
            hr(),
            actionButton("load_sampled_data_btn", "Load Sampled Data", class = "btn-info btn-block", icon = icon("refresh")),
            actionButton("consider_scenario_btn", "Consider Scenario", class = "btn-success btn-block", icon = icon("save"), disabled = TRUE),
            hr(),
            DT::dataTableOutput("offers_table")
          )
        )
      ),
      
      # --- TAB 2: STRATEGY SELECTION ---
      tabItem(tabName = "strategy",
        fluidRow(
          # --- UI: Place Bandit Simulation and Strategy Configuration side by side at the top ---
          # In tabName = 'strategy', replace the top layout with:
          fluidRow(
            column(6,
              box(
                title = "Strategy Configuration", status = "primary", solidHeader = TRUE, width = 12,
                
                h4("Market Conditions Analysis"),
                actionButton("calculate_market_conditions_btn", "Calculate Market Conditions", 
                            class = "btn-info btn-block", icon = icon("calculator")),
                
                hr(),
                
                div(class = "metric-box",
                    div(class = "metric-value", textOutput("market_demand_index")),
                    div(class = "metric-label", "Market Demand Index")
                ),
                
                div(class = "metric-box",
                    div(class = "metric-value", textOutput("demand_category")),
                    div(class = "metric-label", "Demand Category")
                ),
                
                # Removed Market Demand Override and Price Sensitivity Override sliders
                # - Market demand is auto-detected from the four market signals
                # - Price sensitivity is calculated per user based on their profile
                
                hr(),
                
                h4("Strategy Selection"),
                selectInput("ranking_strategy", "Ranking Strategy:",
                           choices = c("Greedy (Highest Commission)" = "Greedy (Highest Commission)",
                                     "User-First (Lowest Price)" = "User-First (Lowest Price)",
                                     "Stochastic LP" = "Stochastic LP",
                                     "RL Optimized Policy" = "RL Optimized Policy")),
                
                conditionalPanel(
                  condition = "input.ranking_strategy == 'Stochastic LP'",
                  h5("Optimization Weights:"),
                  sliderInput("weight_conversion", "Conversion Weight:", 
                             min = 0, max = 1, value = 0.4, step = 0.1),
                  sliderInput("weight_revenue", "Revenue Weight:", 
                             min = 0, max = 1, value = 0.4, step = 0.1),
                  sliderInput("weight_trust", "Trust Weight:", 
                             min = 0, max = 1, value = 0.2, step = 0.1)
                ),
                
                hr(),
                
                actionButton("apply_strategy_btn", "Apply Strategy", 
                            class = "btn-primary btn-block", icon = icon("cogs"))
              )
            ),
            column(6,
              box(
                title = "Bandit Simulation: Ranking Impact Over Clicks", status = "warning", solidHeader = TRUE, width = 12,
                fluidRow(
                  column(6, actionButton("run_bandit_sim_btn", "Estimate ranking impact over clicks", class = "btn-warning btn-lg")),
                  column(6, div(style = "margin-top: 25px;", textOutput("bandit_sim_status")))
                ),
                DT::dataTableOutput("bandit_sim_results_table"),
                verbatimTextOutput("bandit_sim_summary"),
                plotlyOutput("bandit_sim_barplot")
              )
            )
          ),
          # The merged User, Market State & Offers Table box remains below this row
          box(
            title = "User, Market State & Offers Table", status = "info", solidHeader = TRUE, width = 12,
            fluidRow(
              column(4, selectizeInput("selected_users_for_offers", "Select User ID(s):", choices = NULL, multiple = TRUE, width = "100%")),
              column(8, div(style = "margin-top: 25px;", textOutput("offers_loading_status")))
            ),
            tags$div(
              style = "margin-bottom: 10px;",
              tags$span("Dynamic Price Sensitivity: ", title = "A user-specific metric combining base price sensitivity, days to go, and market volatility. See README for formula."),
              tags$span(" | "),
              tags$span("Market Demand Index: ", title = "Composite index of price, urgency, volatility, and competition. See README for formula.")
            ),
            DT::dataTableOutput("merged_offers_table")
          )
        )
      ),
      
      # --- TAB 3: RESULTS & ANALYSIS ---
      tabItem(tabName = "results",
        fluidRow(
          # Performance Metrics
          box(
            title = "Performance Metrics", status = "success", solidHeader = TRUE, width = 12,
            fluidRow(
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("total_revenue")),
                           div(class = "metric-label", "Total Revenue"))),
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("avg_trust")),
                           div(class = "metric-label", "Avg Trust Score"))),
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("conversion_rate")),
                           div(class = "metric-label", "Conversion Rate"))),
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("click_through_rate")),
                           div(class = "metric-label", "Click-Through Rate"))),
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("price_consistency")),
                           div(class = "metric-label", "Price Consistency"))),
              column(2, div(class = "metric-box",
                           div(class = "metric-value", textOutput("profit_margin")),
                           div(class = "metric-label", "Profit Margin")))
            )
          )
        ),
        
        fluidRow(
          # Ranked Hotels List
          box(
            title = "Ranked Hotel Offers", status = "primary", solidHeader = TRUE, width = 12,
            DT::dataTableOutput("ranked_offers_table")
          )
        ),
        
        fluidRow(
          # Revenue vs Trust Plot
          box(
            title = "Revenue vs User Trust Analysis", status = "info", solidHeader = TRUE, width = 6,
            plotlyOutput("revenue_trust_plot")
          ),
          
          # Multi-Armed Bandit Simulation
          box(
            title = "Strategy Comparison (MAB Simulation)", status = "warning", solidHeader = TRUE, width = 6,
            actionButton("run_mab_btn", "Run MAB Simulation", class = "btn-warning"),
            br(), br(),
            plotlyOutput("mab_simulation_plot")
          )
        )
      ),
      
      # --- TAB 4: MARKET ANALYSIS ---
      tabItem(tabName = "market",
        fluidRow(
          box(
            title = "Market Intelligence Dashboard", status = "primary", solidHeader = TRUE, width = 12,
            
            h4("Destination Market Overview"),
            verbatimTextOutput("market_overview"),
            
            h4("Price Trends"),
            plotlyOutput("price_trends_plot"),
            
            h4("Competitor Analysis"),
            DT::dataTableOutput("competitor_analysis_table")
          )
        )
      ),
      tabItem(tabName = "generated_data",
        box(title = "Generated Data Overview", status = "primary", solidHeader = TRUE, width = 12,
          DT::dataTableOutput("generated_data_table")
        )
      )
    )
  )
)

# --- Server Logic ---
server <- function(input, output, session) {
  
  # Debug startup
  print("=== SERVER FUNCTION STARTING ===")
  print("Server function initialized")
  write(paste("Server starting at", Sys.time()), "/tmp/shiny_debug.log", append = TRUE)
  
  # Initialize reactive values
  values <- reactiveValues(
    user_offers = NULL,
    scenario_saved = FALSE,
    user_ids = NULL,
    selected_user_profile = NULL,
    data_loaded = FALSE,  # Track if data has been loaded
    market_conditions = NULL,  # Store market conditions analysis
    user_market_table = NULL, # New reactive value for user/market table
    offers_prob_table = NULL, # New reactive value for offers/probabilities table
    bandit_sim_table = NULL, # New reactive value for bandit simulation results
    bandit_sim_summary = NULL, # New reactive value for bandit simulation summary
    bandit_sim_top = NULL, # New reactive value for top arms for barplot
    user_offers_multi = NULL, # New reactive value for offers table with multi-user selection
    bandit_results = NULL, # New reactive value for bandit simulation results
    merged_offers = NULL, # New reactive value for merged offers table
    bandit_sim_results_table = NULL, # New reactive value for bandit simulation results table
    bandit_simulation_from_csv = NULL
  )

  # Define fetch_and_update_user_ids but do NOT call it at the top level
  fetch_and_update_user_ids <- function() {
    tryCatch({
      if (!is.null(values$user_offers) && length(values$user_offers) > 0) {
        df <- as.data.frame(values$user_offers, stringsAsFactors = FALSE)
        if ("user_id" %in% colnames(df)) {
          user_ids <- unique(df$user_id)
          user_ids <- user_ids[!is.na(user_ids) & user_ids != ""]
          if (length(user_ids) > 0) {
            values$user_ids <- user_ids
            updateSelectInput(session, "user_profile_select_offers", choices = user_ids)
            updateSelectizeInput(session, "selected_users_for_offers", choices = user_ids)
            print(paste("Updated user IDs from offers data:", length(user_ids), "users"))
            write(paste("Updated user IDs from offers data:", length(user_ids), "users"), "/tmp/shiny_debug.log", append = TRUE)
          } else {
            values$user_ids <- NULL
            updateSelectInput(session, "user_profile_select_offers", choices = NULL)
            updateSelectizeInput(session, "selected_users_for_offers", choices = NULL)
            print("No valid user IDs found in offers data")
            write("No valid user IDs found in offers data", "/tmp/shiny_debug.log", append = TRUE)
          }
        } else {
          values$user_ids <- NULL
          updateSelectInput(session, "user_profile_select_offers", choices = NULL)
          updateSelectizeInput(session, "selected_users_for_offers", choices = NULL)
          print("No user_id column found in offers data")
          write("No user_id column found in offers data", "/tmp/shiny_debug.log", append = TRUE)
        }
      } else {
        values$user_ids <- NULL
        updateSelectInput(session, "user_profile_select_offers", choices = NULL)
        updateSelectizeInput(session, "selected_users_for_offers", choices = NULL)
        print("No offers data available to extract user IDs")
        write("No offers data available to extract user IDs", "/tmp/shiny_debug.log", append = TRUE)
      }
    }, error = function(e) {
      values$user_ids <- NULL
      updateSelectInput(session, "user_profile_select_offers", choices = NULL)
      updateSelectizeInput(session, "selected_users_for_offers", choices = NULL)
      print(paste("Error updating user IDs:", e$message))
      write(paste("Error updating user IDs:", e$message), "/tmp/shiny_debug.log", append = TRUE)
    })
  }

  # Helper to fetch and update offers table from /trial_sampled_offers
  fetch_and_update_offers <- function() {
    tryCatch({
      print("=== FETCH_AND_UPDATE_OFFERS CALLED ===")
      write("fetch_and_update_offers called", "/tmp/shiny_debug.log", append = TRUE)
      
      res <- GET(paste0(API_URL, "/trial_sampled_offers"))
      print(paste("Response status:", http_status(res)$category))
      
      if (http_status(res)$category == "Success") {
        result <- fromJSON(content(res, "text", encoding = "UTF-8"))
        print(paste("Received", length(result$data), "records"))
        write(paste("Received", length(result$data), "records"), "/tmp/shiny_debug.log", append = TRUE)
        
        if (length(result$data) > 0) {
          values$user_offers <- result$data
          print("Updated values$user_offers successfully")
          write("Updated values$user_offers successfully", "/tmp/shiny_debug.log", append = TRUE)
          showNotification(paste("Offers table updated. Records:", result$records), type = "default")
          # Update user IDs from the offers data
          fetch_and_update_user_ids()
        } else {
          values$user_offers <- NULL
          print("No data in response, set values$user_offers to NULL")
          write("No data in response, set values$user_offers to NULL", "/tmp/shiny_debug.log", append = TRUE)
          showNotification("No offers data found", type = "warning")
          # Clear user IDs since no offers data
          fetch_and_update_user_ids()
        }
      } else {
        print("HTTP request failed")
        write("HTTP request failed", "/tmp/shiny_debug.log", append = TRUE)
        showNotification("Error loading offers table from backend", type = "error")
      }
    }, error = function(e) {
      print(paste("Error in fetch_and_update_offers:", e$message))
      write(paste("Error in fetch_and_update_offers:", e$message), "/tmp/shiny_debug.log", append = TRUE)
      showNotification(paste("Error loading offers table:", e$message), type = "error")
    })
  }

  # On app load, fetch the offers table and load bandit results from CSV
  fetch_and_update_offers()

  # Robust error handling for the merged generated data table
  output$generated_data_table <- DT::renderDataTable({
    user_ids <- input$selected_users_for_offers
    data <- tryCatch({
      # Check for file existence before reading
      if (!file.exists("/data/market_state_by_location.csv") ||
          !file.exists("/data/bandit_simulation_results.csv") ||
          !file.exists("/data/conversion_probabilities.csv")) {
        showNotification("One or more required data files are missing.", type = "error")
        return(NULL)
      }
      load_generated_data(user_ids)
    }, error = function(e) {
      showNotification("Error loading generated data table.", type = "error")
      return(NULL)
    })
    if (is.null(data) || nrow(data) == 0) {
      showNotification("No generated data available for the selected user(s).", type = "warning")
      return(NULL)
    }
    DT::datatable(data, options = list(pageLength = 20, scrollX = TRUE), rownames = FALSE)
  })

  # Load bandit simulation results from CSV on startup
  tryCatch({
    bandit_results_df <- read.csv("../data/bandit_simulation_results.csv")
    values$bandit_simulation_from_csv <- bandit_results_df
    print("Successfully loaded bandit_simulation_results.csv")
  }, error = function(e) {
    print(paste("Error loading bandit_simulation_results.csv:", e$message))
    # Do not showNotification here
  })

  # Render the loaded bandit simulation results in the UI
  output$bandit_sim_results_table <- DT::renderDataTable({
    if (!is.null(values$bandit_simulation_from_csv)) {
      DT::datatable(values$bandit_simulation_from_csv, 
                    options = list(pageLength = 5, scrollX = TRUE, autoWidth = TRUE),
                    rownames = FALSE,
                    caption = 'Bandit Simulation Results from CSV')
    } else {
      DT::datatable(data.frame(Status=character()), caption = 'No bandit simulation data loaded from file.')
    }
  })

  # Test button handler
  observeEvent(input$test_btn, {
    print("Test button clicked!")
    write("Test button clicked", "/tmp/shiny_debug.log", append = TRUE)
    showNotification("App is working! Test button clicked.", type = "default")
  })
  
  # When user selection changes, fetch user profile from backend
  observeEvent(input$user_profile_select_offers, {
    user_id <- input$user_profile_select_offers
    if (!is.null(user_id) && user_id != "") {
      tryCatch({
        res <- GET(paste0(API_URL, "/user_profile/", user_id))
        if (http_status(res)$category == "Success") {
          result <- fromJSON(content(res, "text", encoding = "UTF-8"))
          values$selected_user_profile <- result$profile
        } else {
          values$selected_user_profile <- NULL
        }
      }, error = function(e) {
        values$selected_user_profile <- NULL
      })
    } else {
      values$selected_user_profile <- NULL
    }
  })

  # Render user profile info
  output$user_profile_display <- renderUI({
    profile <- values$selected_user_profile
    if (is.null(profile)) {
      return(HTML("<em>No user selected or profile not found.</em>"))
    }
    # Display user profile fields as a list
    tags$ul(
      lapply(names(profile), function(field) {
        tags$li(strong(field), ": ", as.character(profile[[field]]))
      })
    )
  })

  # Enable/disable buttons based on scenario state
  observe({
    if (values$scenario_saved) {
      shinyjs::disable("load_sampled_data_btn")
      shinyjs::disable("consider_scenario_btn")
      shinyjs::disable("num_hotels")
      shinyjs::disable("num_partners")
      shinyjs::disable("days_to_go")
      shinyjs::disable("days_var")
      shinyjs::disable("num_users")
    } else if (values$data_loaded) {
      shinyjs::disable("load_sampled_data_btn")
      shinyjs::enable("consider_scenario_btn")
      shinyjs::disable("num_hotels")
      shinyjs::disable("num_partners")
      shinyjs::disable("days_to_go")
      shinyjs::disable("days_var")
      shinyjs::disable("num_users")
    } else {
      shinyjs::enable("load_sampled_data_btn")
      shinyjs::disable("consider_scenario_btn")
      shinyjs::enable("num_hotels")
      shinyjs::enable("num_partners")
      shinyjs::enable("days_to_go")
      shinyjs::enable("days_var")
      shinyjs::enable("num_users")
    }
  })

  # Load Sampled Data button handler
  observeEvent(input$load_sampled_data_btn, {
    # Call backend to sample new data for multiple users
    num_users <- input$num_users
    num_hotels <- input$num_hotels
    num_partners <- input$num_partners
    days_to_go <- input$days_to_go
    days_var <- input$days_var
    url <- paste0(API_URL, "/sample_offers_for_users",
                 "?num_users=", num_users,
                 "&num_hotels=", num_hotels,
                 "&num_partners=", num_partners,
                 "&days_to_go=", days_to_go,
                 "&days_var=", days_var)
    res <- POST(url)
    # After backend samples, reload offers (user IDs will be updated automatically)
    fetch_and_update_offers()
    
    # Explicitly update the selectizeInput for selected_users_for_offers
    # This ensures the dropdown is populated with the correct user IDs from the sampled data
    if (!is.null(values$user_ids) && length(values$user_ids) > 0) {
      updateSelectizeInput(session, "selected_users_for_offers", choices = values$user_ids, server = TRUE)
      print(paste("Updated selected_users_for_offers with", length(values$user_ids), "user IDs after sampling"))
      write(paste("Updated selected_users_for_offers with", length(values$user_ids), "user IDs after sampling"), "/tmp/shiny_debug.log", append = TRUE)
    }
    
    # Set data loaded flag
    values$data_loaded <- TRUE
  })

  # Render offers table: show all columns, filtered by selected user, with correct headers
  output$offers_table <- DT::renderDataTable({
    if (is.null(values$user_offers)) return(NULL)
    df <- as.data.frame(values$user_offers, stringsAsFactors = FALSE)
    user_id <- input$user_profile_select_offers
    if (!is.null(user_id) && user_id != "") {
      df <- df[df$user_id == user_id, , drop = FALSE]
    }
    DT::datatable(
      df,
      options = list(
        scrollX = TRUE,  # Enable horizontal scrolling
        scrollY = "400px",  # Enable vertical scrolling with fixed height
        pageLength = 25,  # Show 25 rows per page
        lengthMenu = c(10, 25, 50, 100),  # Page length options
        autoWidth = FALSE,  # Don't auto-adjust column widths
        columnDefs = list(
          list(targets = "_all", className = "dt-center")  # Center align all columns
        )
      ),
      rownames = FALSE,  # Don't show row numbers
      filter = "top",  # Add filter boxes at the top
      selection = "single"  # Allow single row selection
    )
  })

  # --- Automatically generate conversion_probabilities.csv after 'Consider Scenario' ---
  observeEvent(input$consider_scenario_btn, {
    print("Consider Scenario button clicked")
    print(paste("fresh_offers is null:", is.null(values$user_offers)))
    if (!is.null(values$user_offers)) {
      print(paste("fresh_offers length:", length(values$user_offers)))
    }
    write("Consider Scenario button clicked", "/tmp/shiny_debug.log", append = TRUE)
    if (is.null(values$user_offers) || length(values$user_offers) == 0) {
      showNotification("No offers data available to save! Please load sample profiles first.", type = "warning")
      write("No fresh offers data available to save!", "/tmp/shiny_debug.log", append = TRUE)
      return()
    }
    showNotification("Saving scenario data...", id = "save_scenario", duration = NULL, type = "message")
    tryCatch({
      scenario_data <- list(all_offers = values$user_offers)
      print(paste("Saving", length(values$user_offers), "offers"))
      write(paste("Saving", length(values$user_offers), "offers"), "/tmp/shiny_debug.log", append = TRUE)
      res <- POST(
        url = paste0(API_URL, "/save_scenario"),
        body = toJSON(scenario_data, auto_unbox = TRUE),
        encode = "json",
        add_headers("Content-Type" = "application/json")
      )
      if (http_status(res)$category == "Success") {
        result <- fromJSON(content(res, "text", encoding = "UTF-8"))
        removeNotification("save_scenario")
        showNotification(
          paste("Scenario saved successfully! File:", result$filename, "Records:", result$records_saved), 
          type = "default"
        )
        write(paste("Scenario saved successfully! File:", result$filename), "/tmp/shiny_debug.log", append = TRUE)
        
        # Set the scenario saved flag
        values$scenario_saved <- TRUE
        
        # Fetch the updated offers table from backend to show the saved data (user IDs will be updated automatically)
        fetch_and_update_offers()
        
        # Explicitly update the selectizeInput for selected_users_for_offers
        # This ensures the dropdown is populated with the correct user IDs from the saved scenario
        if (!is.null(values$user_ids) && length(values$user_ids) > 0) {
          updateSelectizeInput(session, "selected_users_for_offers", choices = values$user_ids, server = TRUE)
          print(paste("Updated selected_users_for_offers with", length(values$user_ids), "user IDs after scenario save"))
          write(paste("Updated selected_users_for_offers with", length(values$user_ids), "user IDs after scenario save"), "/tmp/shiny_debug.log", append = TRUE)
        }
        
      } else {
        removeNotification("save_scenario")
        showNotification("Error saving scenario data", type = "error")
        write("Error saving scenario data", "/tmp/shiny_debug.log", append = TRUE)
      }
    }, error = function(e) {
      removeNotification("save_scenario")
      showNotification(paste("Error saving scenario:", e$message), type = "error")
      write(paste("Error saving scenario:", e$message), "/tmp/shiny_debug.log", append = TRUE)
    })
    # After scenario is saved, trigger conversion probabilities CSV generation
    res <- POST(paste0(API_URL, "/conversion_probabilities_csv"))
    if (http_status(res)$category == "Success") {
      showNotification("Conversion probabilities CSV generated.", type = "message")
    } else {
      showNotification("Failed to generate conversion probabilities CSV.", type = "error")
    }
  })

  # --- TAB 2: STRATEGY SELECTION ---
  
  # Calculate Market Conditions button handler
  observeEvent(input$calculate_market_conditions_btn, {
    showNotification("Calculating market conditions...", id = "market_calc", duration = NULL, type = "message")
    
    tryCatch({
      res <- POST(paste0(API_URL, "/calculate_market_conditions"))
      
      if (http_status(res)$category == "Success") {
        result <- fromJSON(content(res, "text", encoding = "UTF-8"))
        values$market_conditions <- result
        
        # Auto-update the market demand override if set to auto
        if (input$market_demand_override == "auto") {
          updateSelectInput(session, "market_demand_override", selected = result$demand_category)
        }
        
        removeNotification("market_calc")
        showNotification("Market conditions calculated successfully!", type = "default")
      } else {
        removeNotification("market_calc")
        showNotification("Error calculating market conditions", type = "error")
      }
    }, error = function(e) {
      removeNotification("market_calc")
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Market Demand Index output
  output$market_demand_index <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$market_demand_index
  })
  
  # Demand Category output
  output$demand_category <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$demand_category
  })
  
  # Price Level outputs
  output$price_level_value <- renderText({
    if (is.null(values$market_conditions)) return("--")
    paste0("$", values$market_conditions$signals$price_level$value)
  })
  
  output$price_level_norm <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$price_level$normalized
  })
  
  # Booking Urgency outputs
  output$booking_urgency_value <- renderText({
    if (is.null(values$market_conditions)) return("--")
    paste0(values$market_conditions$signals$booking_urgency$value, " days")
  })
  
  output$booking_urgency_norm <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$booking_urgency$normalized
  })
  
  # Price Volatility outputs
  output$price_volatility_value <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$price_volatility$value
  })
  
  output$price_volatility_norm <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$price_volatility$normalized
  })
  
  # Price Trend outputs
  output$price_trend_value <- renderText({
    if (is.null(values$market_conditions)) return("--")
    paste0(values$market_conditions$signals$price_trend$value, "%")
  })
  
  output$price_trend_direction <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$price_trend$direction
  })
  
  # Competition Density outputs
  output$competition_density_value <- renderText({
    if (is.null(values$market_conditions)) return("--")
    paste0(values$market_conditions$signals$competition_density$unique_hotels, " × ", 
           values$market_conditions$signals$competition_density$unique_partners)
  })
  
  output$competition_density_norm <- renderText({
    if (is.null(values$market_conditions)) return("--")
    values$market_conditions$signals$competition_density$normalized
  })
  
  # Market Insights output
  output$market_insights <- renderText({
    if (is.null(values$market_conditions)) return("No market data available")
    
    insights <- values$market_conditions$market_insights
    summary <- values$market_conditions$data_summary
    
          paste(
        "Market Demand Index:", values$market_conditions$market_demand_index,
        "\nDemand Category:", values$market_conditions$demand_category,
        "\n\nMarket Insights:",
        "\n• Price Trend:", insights$price_trend,
        "\n• Price Volatility:", insights$price_volatility,
        "\n• Competition Level:", insights$competition_level,
        "\n• Booking Pressure:", insights$booking_pressure,
      "\n\nData Summary:",
      "\n• Total Offers:", summary$total_offers,
      "\n• Unique Hotels:", summary$unique_hotels,
      "\n• Unique Partners:", summary$unique_partners,
      "\n• Price Range:", summary$price_range,
      "\n• Days Range:", summary$days_range
    )
  })
  
  # Apply strategy
  observeEvent(input$apply_strategy_btn, {
    if (is.null(values$scenario_data)) {
      showNotification("Please generate a scenario first!", type = "warning")
      return()
    }
    
    showNotification("Applying ranking strategy...", id = "strategy_apply", duration = NULL, type = "message")
    
    tryCatch({
      # Generate customer behavior
      customer_behavior_res <- POST(
        url = paste0(API_URL, "/generate_scenario"),
        body = toJSON(list(
          user_profile = values$scenario_data$user_profile,
          destination = values$scenario_data$market_conditions$destination,
          num_hotels = 1,
          num_partners = 1
        ), auto_unbox = TRUE),
        encode = "json",
        add_headers("Content-Type" = "application/json")
      )
      
      # Prepare strategy config
      strategy_config <- list(
        strategy_name = input$ranking_strategy,
        optimization_method = "pulp",
        objective_weights = list(),
        constraints = list(),
        bandit_algorithm = "epsilon_greedy",
        exploration_rate = 0.1
      )
      
      if (input$ranking_strategy == "Stochastic LP") {
        strategy_config$objective_weights <- list(
          conversion = input$weight_conversion,
          revenue = input$weight_revenue,
          trust = input$weight_trust
        )
      }
      
      # Mock customer behavior (simplified for demo)
      customer_behavior <- list(
        user_id = values$scenario_data$user_profile$user_id,
        price_sensitivity_calculated = input$price_sensitivity_override,
        conversion_likelihood = 0.6,
        brand_preference = list(
          "Booking.com" = 0.8,
          "Expedia" = 0.7,
          "Hotels.com" = 0.6,
          "Agoda" = 0.5,
          "HotelDirect" = 0.4
        ),
        amenity_importance = list(),
        location_importance = 0.7,
        review_sensitivity = 0.8,
        cancellation_preference = "Flexible",
        booking_urgency = max(0, 1 - (values$scenario_data$market_conditions$days_until_travel / 180))
      )
      
      # Prepare ranking request
      ranking_request <- list(
        scenario = values$scenario_data,
        strategy_config = strategy_config,
        customer_behavior = customer_behavior
      )
      
      res <- POST(
        url = paste0(API_URL, "/rank_offers"),
        body = toJSON(ranking_request, auto_unbox = TRUE),
        encode = "json",
        add_headers("Content-Type" = "application/json")
      )
      
      if (http_status(res)$category == "Success") {
        ranking_data <- fromJSON(content(res, "text", encoding = "UTF-8"))
        values$ranking_results <- ranking_data
        values$customer_behavior <- customer_behavior
        
        removeNotification("strategy_apply")
        showNotification("Strategy applied successfully!", type = "default")
      } else {
        removeNotification("strategy_apply")
        showNotification("Error applying strategy", type = "error")
      }
    }, error = function(e) {
      removeNotification("strategy_apply")
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Customer behavior predictions
  output$predicted_conversion <- renderText({
    if (is.null(values$customer_behavior)) return("--")
    paste0(round(values$customer_behavior$conversion_likelihood * 100, 1), "%")
  })
  
  output$booking_urgency <- renderText({
    if (is.null(values$customer_behavior)) return("--")
    paste0(round(values$customer_behavior$booking_urgency * 100, 1), "%")
  })
  
  output$price_sensitivity_calc <- renderText({
    if (is.null(values$customer_behavior)) return("--")
    paste0(round(values$customer_behavior$price_sensitivity_calculated * 100, 1), "%")
  })
  
  output$brand_preference <- renderText({
    if (is.null(values$customer_behavior)) return("--")
    top_brand <- names(which.max(unlist(values$customer_behavior$brand_preference)))
    top_brand
  })
  
  # --- TAB 3: RESULTS & ANALYSIS ---
  
  # Performance metrics
  output$total_revenue <- renderText({
    if (is.null(values$ranking_results)) return("--")
    format_currency(values$ranking_results$performance_metrics$total_expected_revenue)
  })
  
  output$avg_trust <- renderText({
    if (is.null(values$ranking_results)) return("--")
    paste0(round(values$ranking_results$performance_metrics$average_user_trust, 1), "%")
  })
  
  output$conversion_rate <- renderText({
    if (is.null(values$ranking_results)) return("--")
    paste0(round(values$ranking_results$performance_metrics$conversion_rate * 100, 1), "%")
  })
  
  output$click_through_rate <- renderText({
    if (is.null(values$ranking_results)) return("--")
    paste0(round(values$ranking_results$performance_metrics$click_through_rate * 100, 1), "%")
  })
  
  output$price_consistency <- renderText({
    if (is.null(values$ranking_results)) return("--")
    paste0(round(values$ranking_results$performance_metrics$price_consistency * 100, 1), "%")
  })
  
  output$profit_margin <- renderText({
    if (is.null(values$ranking_results)) return("--")
    paste0(round(values$ranking_results$performance_metrics$profit_margin * 100, 1), "%")
  })
  
  # Ranked offers table
  output$ranked_offers_table <- DT::renderDataTable({
    if (is.null(values$ranking_results)) return(NULL)
    
    ranked_df <- do.call(rbind, lapply(values$ranking_results$ranked_offers, function(r) {
      offer <- r$offer
      # Find corresponding hotel
      hotel <- values$scenario_data$hotels[[which(sapply(values$scenario_data$hotels, function(h) h$hotel_id) == offer$hotel_id)]]
      
      data.frame(
        Rank = r$rank,
        Hotel = hotel$name,
        Partner = offer$partner_name,
        Price = offer$price_per_night,
        Score = round(r$score, 2),
        Conversion_Prob = paste0(round(r$conversion_probability * 100, 1), "%"),
        Expected_Revenue = r$expected_revenue,
        Trust_Score = round(r$user_trust_score, 1),
        Price_Consistency = paste0(round(r$price_consistency_score * 100, 1), "%"),
        Explanation = r$explanation,
        stringsAsFactors = FALSE
      )
    }))
    
    datatable(ranked_df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
      formatCurrency(columns = c("Price", "Expected_Revenue"), currency = "$") %>%
      formatStyle("Rank", backgroundColor = styleInterval(c(1, 3, 5), c("#d4edda", "#fff3cd", "#f8d7da", "#ffffff")))
  })
  
  # Revenue vs Trust plot
  output$revenue_trust_plot <- renderPlotly({
    if (is.null(values$ranking_results)) {
      p <- ggplot() + 
        geom_text(aes(x = 0.5, y = 0.5, label = "No data available"), size = 6) +
        theme_void()
      return(ggplotly(p))
    }
    
    plot_data <- do.call(rbind, lapply(values$ranking_results$ranked_offers, function(r) {
      data.frame(
        Revenue = r$expected_revenue,
        Trust = r$user_trust_score,
        Rank = r$rank,
        Hotel = r$offer$partner_name,
        Price = r$offer$price_per_night
      )
    }))
    
    p <- ggplot(plot_data, aes(x = Trust, y = Revenue, color = factor(Rank), size = Price)) +
      geom_point(alpha = 0.7) +
      geom_text(aes(label = Rank), vjust = -0.5, size = 3) +
      scale_color_viridis_d(name = "Rank") +
      scale_size_continuous(name = "Price", range = c(3, 8)) +
      labs(
        title = "Revenue vs User Trust Trade-off",
        x = "User Trust Score",
        y = "Expected Revenue ($)",
        subtitle = "Bubble size represents price per night"
      ) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # MAB Simulation
  observeEvent(input$run_mab_btn, {
    if (is.null(values$scenario_data)) {
      showNotification("Please generate a scenario first!", type = "warning")
      return()
    }
    
    showNotification("Running MAB simulation...", id = "mab_sim", duration = NULL, type = "message")
    
    tryCatch({
      mab_request <- list(
        strategies = c("Greedy (Highest Commission)", "User-First (Lowest Price)", 
                      "Stochastic LP", "RL Optimized Policy"),
        num_iterations = 100,
        scenario = values$scenario_data
      )
      
      res <- POST(
        url = paste0(API_URL, "/mab_simulation"),
        body = toJSON(mab_request, auto_unbox = TRUE),
        encode = "json",
        add_headers("Content-Type" = "application/json")
      )
      
      if (http_status(res)$category == "Success") {
        mab_data <- fromJSON(content(res, "text", encoding = "UTF-8"))
        values$mab_results <- mab_data
        
        removeNotification("mab_sim")
        showNotification("MAB simulation completed!", type = "default")
      } else {
        removeNotification("mab_sim")
        showNotification("Error running MAB simulation", type = "error")
      }
    }, error = function(e) {
      removeNotification("mab_sim")
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # MAB simulation plot
  output$bandit_sim_results_table <- DT::renderDataTable({
    req(values$bandit_sim_results_table)
    DT::datatable(values$bandit_sim_results_table, options = list(scrollX = TRUE, pageLength = 10), rownames = FALSE)
  })

  output$mab_simulation_plot <- renderPlotly({
    if (is.null(values$mab_results)) {
      p <- ggplot() + 
        geom_text(aes(x = 0.5, y = 0.5, label = "Click 'Run MAB Simulation' to see results"), size = 4) +
        theme_void()
      return(ggplotly(p))
    }
    
    plot_data <- data.frame(
      Iteration = 1:length(values$mab_results$cumulative_rewards),
      Cumulative_Reward = values$mab_results$cumulative_rewards
    )
    
    p <- ggplot(plot_data, aes(x = Iteration, y = Cumulative_Reward)) +
      geom_line(color = "#2c3e50", size = 1.2) +
      geom_point(color = "#e74c3c", alpha = 0.6) +
      labs(
        title = "Multi-Armed Bandit Strategy Comparison",
        subtitle = paste("Best Strategy:", values$mab_results$best_strategy),
        x = "Iteration",
        y = "Cumulative Reward"
      ) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # --- TAB 4: MARKET ANALYSIS ---
  
  # Market overview (placeholder)
  output$market_overview <- renderText({
    if (is.null(values$scenario_data)) return("No market data available")
    
    paste(
      "Market Intelligence Dashboard",
      "========================",
      paste("Destination:", values$scenario_data$market_conditions$destination),
      paste("Current Demand Level:", values$scenario_data$market_conditions$market_demand),
      paste("Seasonal Factor:", round(values$scenario_data$market_conditions$seasonal_factor, 2)),
      paste("Market Volatility:", round(values$scenario_data$market_conditions$price_volatility, 2)),
      "",
      "Key Insights:",
      "- Monitor competitor pricing closely",
      "- Adjust strategies based on demand patterns",
      "- Consider seasonal pricing adjustments",
      sep = "\n"
    )
  })
  
  # Price trends plot using real data from sampled offers
  output$price_trends_plot <- renderPlotly({
    if (is.null(values$user_offers) || length(values$user_offers) == 0) {
      p <- ggplot() + 
        geom_text(aes(x = 0.5, y = 0.5, label = "Load sampled data to see price trends"), size = 4) +
        theme_void()
      return(ggplotly(p))
    }
    
    # Use the sampled offers data to show price history trends
    price_history_data <- data.frame()
    
    for (offer in values$user_offers) {
      tryCatch({
        if (!is.null(offer$price_history_24h)) {
          # Parse the 24-hour price history
          price_history <- fromJSON(offer$price_history_24h)
          if (length(price_history) == 24) {
            history_df <- data.frame(
              Hour = 0:23,
              Price = price_history,
              Hotel = offer$hotel_name,
              Partner = offer$partner_name
            )
            price_history_data <- rbind(price_history_data, history_df)
          }
        }
      }, error = function(e) {
        # Skip if parsing fails
      })
    }
    
    if (nrow(price_history_data) == 0) {
      # Fallback to price distribution if no price history available
      offers_df <- do.call(rbind, lapply(values$user_offers, function(offer) {
        data.frame(
          Price = offer$price_per_night,
          Hotel = offer$hotel_name,
          Partner = offer$partner_name
        )
      }))
      
      p <- ggplot(offers_df, aes(x = Price)) +
        geom_histogram(bins = 20, fill = "#3498db", alpha = 0.7) +
        geom_vline(aes(xintercept = mean(Price)), color = "#e74c3c", linetype = "dashed", size = 1) +
        labs(
          title = "Price Distribution from Sampled Offers",
          subtitle = paste("Average Price: $", round(mean(offers_df$Price), 2)),
          x = "Price per Night ($)",
          y = "Number of Offers"
        ) +
        theme_minimal()
    } else {
      # Show 24-hour price trends
      p <- ggplot(price_history_data, aes(x = Hour, y = Price, color = Partner)) +
        geom_line(alpha = 0.7) +
        geom_point(alpha = 0.5, size = 1) +
        labs(
          title = "24-Hour Price Trends from Sampled Offers",
          subtitle = paste("Showing", length(unique(price_history_data$Hotel)), "hotels across", 
                          length(unique(price_history_data$Partner)), "partners"),
          x = "Hour (Past 24 Hours)",
          y = "Price ($)",
          color = "Partner"
        ) +
        theme_minimal() +
        theme(legend.position = "bottom")
    }
    
    ggplotly(p)
  })
  
  # Competitor analysis table (placeholder)
  output$competitor_analysis_table <- DT::renderDataTable({
    competitors_df <- data.frame(
      Partner = c("Booking.com", "Expedia", "Hotels.com", "Agoda", "HotelDirect"),
      Market_Share = c("28%", "22%", "18%", "15%", "12%"),
      Avg_Commission = c("15.2%", "13.8%", "14.1%", "12.5%", "16.8%"),
      Price_Competitiveness = c("High", "Medium", "High", "Very High", "Medium"),
      User_Preference = c("9.2", "8.5", "8.1", "7.8", "7.2"),
      stringsAsFactors = FALSE
    )
    
    datatable(competitors_df, options = list(pageLength = 10, dom = 't'), rownames = FALSE)
  })

# Helper to fetch all unique users from trial_sampled_offers
fetch_all_users <- function() {
  res <- GET(paste0(API_URL, "/trial_user_ids"))
  if (http_status(res)$category == "Success") {
    user_ids <- fromJSON(content(res, "text", encoding = "UTF-8"))$user_ids
    return(user_ids)
  } else {
    return(NULL)
  }
}

# Helper to fetch user profile
fetch_user_profile <- function(user_id) {
  res <- GET(paste0(API_URL, "/user_profile/", user_id))
  if (http_status(res)$category == "Success") {
    return(fromJSON(content(res, "text", encoding = "UTF-8"))$profile)
  } else {
    return(NULL)
  }
}

# Helper to fetch dynamic price sensitivity
fetch_dynamic_sensitivity <- function(user_id) {
  res <- GET(paste0(API_URL, "/dynamic_price_sensitivity/", user_id))
  if (http_status(res)$category == "Success") {
    return(fromJSON(content(res, "text", encoding = "UTF-8")))
  } else {
    return(NULL)
  }
}

# Helper to fetch market state
fetch_market_state <- function(location) {
  res <- GET(paste0(API_URL, "/market_state/", URLencode(location)))
  if (http_status(res)$category == "Success") {
    return(fromJSON(content(res, "text", encoding = "UTF-8")))
  } else {
    return(NULL)
  }
}

# --- Conversion Probability Integration ---
# Helper to fetch conversion probability for a user-offer pair
fetch_conversion_probability <- function(user_id, offer_id) {
  res <- GET(paste0(API_URL, "/conversion_probability/", user_id, "/", offer_id))
  if (http_status(res)$category == "Success") {
    prob <- fromJSON(content(res, "text", encoding = "UTF-8"))
    if (!is.null(prob$conversion_probability)) return(prob$conversion_probability)
  }
  return(NA)
}

# Helper to trigger batch CSV generation for conversion probabilities
generate_conversion_probabilities_csv <- function() {
  res <- POST(paste0(API_URL, "/conversion_probabilities_csv"))
  if (http_status(res)$category == "Success") {
    showNotification("Conversion probabilities CSV generated.", type = "message")
  } else {
    showNotification("Failed to generate conversion probabilities CSV.", type = "error")
  }
}

# Add an action button for admin/testing to generate the CSV
addResourcePath("custom", "./")

# Add the button to the UI (e.g., in the sidebar or at the top right)
insertUI(
  selector = "body",
  where = "beforeBegin",
  ui = actionButton("generate_conversion_csv_btn", "Generate Conversion Probabilities CSV", class = "btn-info", style = "position: fixed; top: 60px; right: 10px; z-index: 1001;")
)

# Observe the button click in the server
observeEvent(input$generate_conversion_csv_btn, {
  generate_conversion_probabilities_csv()
})

# Reactive value to store user/market table
values$user_market_table <- NULL

observeEvent(input$refresh_user_market_btn, {
  showModal(modalDialog("Loading user and market data...", footer = NULL))
  user_ids <- fetch_all_users()
  if (is.null(user_ids) || length(user_ids) == 0) {
    values$user_market_table <- NULL
    removeModal()
    return()
  }
  user_rows <- list()
  for (uid in user_ids) {
    profile <- fetch_user_profile(uid)
    dyn_sens <- fetch_dynamic_sensitivity(uid)
    if (is.null(profile) || is.null(dyn_sens)) next
    market_state <- fetch_market_state(dyn_sens$location)
    if (is.null(market_state)) next
    user_rows[[length(user_rows)+1]] <- data.frame(
      UserID = uid,
      Location = dyn_sens$location,
      UserClass = profile$user_type,
      PreferredAmenities = profile$preferred_amenities,
      PriceTrend = round(market_state$price_trend * 100, 2),
      Volatility = round(market_state$price_volatility, 2),
      Competition = market_state$competition_density,
      Urgency = round(market_state$normalized$urgency, 2),
      DemandIndex = round(market_state$demand_index, 2),
      BaseSensitivity = round(dyn_sens$base_price_sensitivity, 2),
      DynamicSensitivity = round(dyn_sens$dynamic_price_sensitivity, 2)
    )
  }
  if (length(user_rows) > 0) {
    values$user_market_table <- do.call(rbind, user_rows)
  } else {
    values$user_market_table <- NULL
  }
  removeModal()
})

# Auto-refresh on app start
observe({
  if (is.null(values$user_market_table)) {
    isolate({
      shinyjs::click("refresh_user_market_btn")
    })
  }
})

# Update user dropdown choices when user_market_table is loaded
observe({
  if (!is.null(values$user_market_table)) {
    updateSelectInput(session, "selected_user_for_offers", choices = values$user_market_table$UserID)
  }
})

# Reactive value to store offers/probabilities for selected user
values$offers_prob_table <- NULL

observeEvent(input$selected_user_for_offers, {
  if (is.null(input$selected_user_for_offers) || input$selected_user_for_offers == "") {
    values$offers_prob_table <- NULL
    output$offers_loading_status <- renderText("")
    return()
  }
  output$offers_loading_status <- renderText("Loading offers and probabilities...")
  user_id <- input$selected_user_for_offers
  res <- GET(paste0(API_URL, "/offer_probabilities/", user_id))
  if (http_status(res)$category == "Success") {
    offers <- fromJSON(content(res, "text", encoding = "UTF-8"))$offers
    if (!is.null(offers) && length(offers) > 0) {
      # Optionally fetch more offer details from trial_sampled_offers.csv if needed
      offers_df <- as.data.frame(offers)
      values$offers_prob_table <- offers_df
      output$offers_loading_status <- renderText("")
    } else {
      values$offers_prob_table <- NULL
      output$offers_loading_status <- renderText("No offers found for this user.")
    }
  } else {
    values$offers_prob_table <- NULL
    output$offers_loading_status <- renderText("Error fetching offers.")
  }
})

output$offers_prob_table <- DT::renderDataTable({
  if (is.null(values$offers_prob_table)) return(NULL)
  datatable(values$offers_prob_table, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
    formatPercentage(c("click_probability", "booking_probability"), 2)
})

values$bandit_sim_table <- NULL
values$bandit_sim_summary <- NULL
values$bandit_sim_top <- NULL

observeEvent(input$run_bandit_sim_btn, {
  output$bandit_sim_status <- renderText("Running bandit simulation, please wait...")
  showModal(modalDialog("Running simulation... this may take a moment.", footer = NULL))

  tryCatch({
    # Step 1: Run the simulation
    res_run <- POST(paste0(API_URL, "/run_bandit_simulation"))
    if (http_status(res_run)$category != "Success") {
      stop("Failed to run bandit simulation on backend.")
    }

    # Step 2: Fetch the results
    res_data <- GET(paste0(API_URL, "/bandit_simulation_results"))
    if (http_status(res_data)$category != "Success") {
      stop("Failed to fetch simulation results.")
    }

    results <- fromJSON(content(res_data, "text", encoding = "UTF-8"))
    if (is.null(results$data) || nrow(as.data.frame(results$data)) == 0) {
      stop("No results returned from simulation.")
    }

    # Data is pivoted, reshape it for ggplot
    df_wide <- as.data.frame(results$data)
    if (!"rank" %in% names(df_wide)) {
      stop("Data from backend is missing 'rank' column.")
    }

    df_long <- tidyr::pivot_longer(df_wide, cols = -rank, names_to = "user_id", values_to = "probability_of_click")
    
    values$bandit_results <- df_long
    values$bandit_sim_results_table <- df_wide # For table view

    # Save the results to a CSV file
    results_path <- file.path("..", "data", "bandit_simulation_results.csv")
    readr::write_csv(df_wide, results_path)
    showNotification("Bandit simulation results have been updated and saved.", type = "message")

    output$bandit_sim_status <- renderText("Simulation complete. Results loaded.")
    showNotification("Bandit simulation results loaded successfully.", type = "message")

  }, error = function(e) {
    output$bandit_sim_status <- renderText(paste("Error:", e$message))
    showNotification(e$message, type = "error")
  }, finally = {
    removeModal()
  })
})

# Add a reactive value to store the CSV data
values$bandit_sim_csv <- NULL

# When the simulation button is clicked, read the CSV directly
observeEvent(input$run_bandit_sim_btn, {
  csv_path <- "/data/bandit_simulation_results.csv"  # Adjust if your mount path is different
  if (file.exists(csv_path)) {
    df <- readr::read_csv(csv_path, col_types = readr::cols(.default = 'c'))
    # Convert numeric columns
    num_cols <- c("rank", "probability_of_click", "true_click_prob", "preference_score", "normalized_probability_of_click")
    for (col in num_cols) {
      if (col %in% names(df)) df[[col]] <- as.numeric(df[[col]])
    }
    values$bandit_sim_csv <- df
    showNotification("Loaded bandit_simulation_results.csv from disk", type = "message")
  } else {
    values$bandit_sim_csv <- NULL
    showNotification("bandit_simulation_results.csv not found on disk", type = "error")
  }
})

# Render the DT table
output$bandit_sim_results_table <- DT::renderDataTable({
  data <- values$bandit_sim_csv 
  if (is.null(data) || !is.data.frame(data) || nrow(data) == 0) {
    return(NULL)
  }
  DT::datatable(data, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
})

output$bandit_sim_table <- DT::renderDataTable({
  if (is.null(values$bandit_sim_table)) return(NULL)
  datatable(values$bandit_sim_table, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
    formatPercentage(c("probability_of_click", "true_click_prob"), 2)
})

output$bandit_sim_summary <- renderPrint({
  if (is.null(values$bandit_sim_summary)) return(NULL)
  s <- values$bandit_sim_summary
  cat(
    "Bandit Simulation Summary:\n",
    "Total Users:", s$total_users, "\n",
    "Total Offers:", s$total_offers, "\n",
    "Total Arms:", s$total_arms, "\n",
    "Clicks per Arm:", s$clicks_per_arm, "\n",
    "CSV Path:", s$csv_path, "\n\n"
  )
  
  if (!is.null(values$bandit_sim_user_sums)) {
    cat("User Probability Sums (should be ~1.0 for each user):\n")
    for (user_id in names(values$bandit_sim_user_sums)) {
      cat(user_id, ":", round(values$bandit_sim_user_sums[[user_id]], 3), "\n")
    }
  }
})

output$bandit_sim_barplot <- renderPlotly({
  if (is.null(values$bandit_results)) return(NULL)
  
  p <- ggplot(values$bandit_results, aes(x = factor(rank), y = probability_of_click, fill = user_id)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(
      title = "Bandit Simulation: Click Probability by Rank",
      x = "Rank",
      y = "Probability of Click",
      fill = "User ID"
    ) +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2")
  
  ggplotly(p)
})

# --- Update Offers & Probabilities for Selected User ---
# 1. Allow multi-select for user IDs
# 2. When users are selected, show offers from trial_sampled_offers.csv filtered by those users
# 3. After simulation, update table with click probabilities, price sensitivity, and market state

# UI: Change selectInput to selectizeInput with multiple=TRUE
observeEvent(input$selected_users_for_offers, {
  user_ids <- input$selected_users_for_offers
  if (is.null(user_ids) || length(user_ids) == 0) {
    values$user_offers_multi <- NULL
    output$offers_loading_status <- renderText({"No user(s) selected"})
    return()
  }
  res <- httr::GET(paste0(API_URL, "/trial_sampled_offers"))
  offers_data <- jsonlite::fromJSON(httr::content(res, as = "text"))
  if (is.null(offers_data$data) || length(offers_data$data) == 0) {
    values$user_offers_multi <- NULL
    output$offers_loading_status <- renderText({"No offers found for selected user(s)"})
    return()
  }
  filtered_offers <- dplyr::filter(as.data.frame(offers_data$data), user_id %in% user_ids)
  values$user_offers_multi <- filtered_offers
  output$offers_loading_status <- renderText({paste(nrow(filtered_offers), "offers loaded for", length(user_ids), "user(s)")})
})

# Render the offers table
output$offers_table_multi <- DT::renderDataTable({
  data <- values$user_offers_multi
  if (is.null(data) || !is.data.frame(data) || nrow(data) == 0) {
    showNotification("No offers data to display.", type = "error")
    return(NULL)
  }
  DT::datatable(data, options = list(pageLength = 10, scrollX = TRUE), server = FALSE)
})

# After simulation, update offers table with click probabilities, price sensitivity, and market state
# (Assume simulation results are loaded into values$bandit_results)
observeEvent(values$bandit_results, {
  if (is.null(values$user_offers_multi) || is.null(values$bandit_results)) return()
  offers <- values$user_offers_multi
  bandit <- values$bandit_results
  # Merge click_probability by user_id and offer_id
  offers <- dplyr::left_join(offers, bandit, by = c("user_id", "offer_id"))
  values$user_offers_multi <- offers
})

# --- Robust checks for empty vectors/data frames in merging logic ---
observeEvent(input$selected_users_for_offers, {
  user_ids <- input$selected_users_for_offers
  if (is.null(user_ids) || length(user_ids) == 0) {
    values$merged_offers <- NULL
    output$offers_loading_status <- renderText({"No user(s) selected"})
    return()
  }
  res <- tryCatch(httr::GET(paste0(API_URL, "/trial_sampled_offers")), error = function(e) NULL)
  if (is.null(res) || httr::status_code(res) != 200) {
    values$merged_offers <- NULL
    output$offers_loading_status <- renderText({"Failed to fetch offers from backend."})
    showNotification("Failed to fetch offers from backend.", type = "error")
    return()
  }
  offers_data <- tryCatch(jsonlite::fromJSON(httr::content(res, as = "text")), error = function(e) NULL)
  if (is.null(offers_data$data) || length(offers_data$data) == 0) {
    values$merged_offers <- NULL
    output$offers_loading_status <- renderText({"No offers found for selected user(s)"})
    showNotification("No offers found for selected user(s)", type = "error")
    return()
  }
  filtered_offers <- dplyr::filter(as.data.frame(offers_data$data), user_id %in% user_ids)
  if (nrow(filtered_offers) == 0) {
    values$merged_offers <- NULL
    output$offers_loading_status <- renderText({"No offers found for selected user(s)"})
    showNotification("No offers found for selected user(s)", type = "error")
    return()
  }
  # For each user, fetch price sensitivity and market state, and append to offers
  merged <- filtered_offers
  merged$price_sensitivity <- NA
  merged$market_state <- NA
  merged$booking_probability <- NA
  for (uid in unique(filtered_offers$user_id)) {
    dps <- tryCatch(jsonlite::fromJSON(httr::content(httr::GET(paste0(API_URL, "/dynamic_price_sensitivity/", uid)), as = "text")), error = function(e) NULL)
    locs <- filtered_offers$location[filtered_offers$user_id == uid]
    if (length(locs) == 0) {
      ms <- NULL
    } else {
      ms <- tryCatch(jsonlite::fromJSON(httr::content(httr::GET(paste0(API_URL, "/market_state/", locs[1])), as = "text")), error = function(e) NULL)
    }
    merged$price_sensitivity[merged$user_id == uid] <- if (!is.null(dps)) dps$dynamic_price_sensitivity else NA
    merged$market_state[merged$user_id == uid] <- if (!is.null(ms)) ms$demand_index else NA
    # Optionally, fetch booking probability from offer_probabilities endpoint
    opp <- tryCatch(jsonlite::fromJSON(httr::content(httr::GET(paste0(API_URL, "/offer_probabilities/", uid)), as = "text")), error = function(e) NULL)
    if (!is.null(opp) && !is.null(opp$offers)) {
      for (i in 1:nrow(merged[merged$user_id == uid,])) {
        oid <- merged$offer_id[merged$user_id == uid][i]
        # Defensive: check opp$offers is a list and oid is not NA
        if (is.na(oid) || length(opp$offers) == 0) next
        idx <- which(sapply(opp$offers, function(x) x$offer_id == oid))
        if (length(idx) == 0) next
        bp <- opp$offers[[idx]]$booking_probability
        merged$booking_probability[merged$user_id == uid & merged$offer_id == oid] <- bp
      }
    }
  }
  values$merged_offers <- merged
  output$offers_loading_status <- renderText({paste(nrow(merged), "offers loaded for", length(user_ids), "user(s)")})
})

output$merged_offers_table <- DT::renderDataTable({
  data <- values$merged_offers
  if (is.null(data) || !is.data.frame(data) || nrow(data) == 0) {
    showNotification("No offers data to display.", type = "error")
    return(NULL)
  }
  df <- as.data.frame(data, stringsAsFactors = FALSE)
  if ("user_id" %in% colnames(df) && "offer_id" %in% colnames(df)) {
    df$conversion_probability <- mapply(fetch_conversion_probability, df$user_id, df$offer_id)
  }
  DT::datatable(df, options = list(pageLength = 10, scrollX = TRUE), server = FALSE)
})

# After simulation, update merged_offers with click probabilities from bandit results
observeEvent(values$bandit_results, {
  if (is.null(values$merged_offers) || is.null(values$bandit_results)) return()
  offers <- values$merged_offers
  bandit <- values$bandit_results
  offers <- dplyr::left_join(offers, bandit, by = c("user_id", "offer_id"))
  values$merged_offers <- offers
})

# Ensure user ID dropdown for offers is populated on app load or after sampling
observe({
  # Use the user IDs from offers data instead of calling a separate endpoint
  if (!is.null(values$user_ids) && length(values$user_ids) > 0) {
    updateSelectizeInput(session, "selected_users_for_offers", choices = values$user_ids, server = TRUE)
  }
})

# --- Add dynamic price sensitivity and market demand index to user/market table ---
# Helper to fetch user dynamic price sensitivity CSV
fetch_user_dynamic_sensitivity_csv <- function() {
  tryCatch({
    res <- httr::POST(paste0(API_URL, "/user_dynamic_price_sensitivity_csv"))
    if (http_status(res)$category == "Success") {
      df <- jsonlite::fromJSON(httr::content(res, as = "text"))
      if (!is.null(df$csv_path) && file.exists(df$csv_path)) {
        return(readr::read_csv(df$csv_path))
      }
    }
    return(NULL)
  }, error = function(e) NULL)
}

# Helper to fetch user market state CSV
fetch_user_market_state_csv <- function() {
  tryCatch({
    path <- "../data/user_market_state.csv"
    if (file.exists(path)) {
      return(readr::read_csv(path))
    }
    return(NULL)
  }, error = function(e) NULL)
}

# Add a reactive value to store the user/market table with new metrics
values$user_market_table_new <- NULL

# On app load or refresh, fetch and merge new metrics
observe({
  dyn_sens_df <- fetch_user_dynamic_sensitivity_csv()
  market_state_df <- fetch_user_market_state_csv()
  if (!is.null(dyn_sens_df) && !is.null(market_state_df)) {
    merged <- dplyr::left_join(dyn_sens_df, market_state_df, by = c("destination" = "location"))
    values$user_market_table_new <- merged
  } else {
    values$user_market_table_new <- NULL
  }
})

# Render the new user/market table
output$user_market_table <- DT::renderDataTable({
  df <- tryCatch(values$user_market_table_new, error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) {
    showNotification("No user/market data available to display.", type = "warning")
    return(NULL)
  }
  DT::datatable(df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
})

# --- Add dynamic price sensitivity, utility, CTR, trust to offers table if available ---
# Helper to fetch dynamic price sensitivity for a user
fetch_dynamic_sensitivity <- function(user_id) {
  res <- GET(paste0(API_URL, "/dynamic_price_sensitivity/", user_id))
  if (http_status(res)$category == "Success") {
    return(fromJSON(content(res, "text", encoding = "UTF-8")))
  } else {
    return(NULL)
  }
}

# Update offers table rendering to include new columns if present
output$offers_table <- DT::renderDataTable({
  if (is.null(values$user_offers)) return(NULL)
  df <- as.data.frame(values$user_offers, stringsAsFactors = FALSE)
  # Add dynamic price sensitivity if available
  if ("user_id" %in% colnames(df)) {
    df$dynamic_price_sensitivity <- sapply(df$user_id, function(uid) {
      dps <- fetch_dynamic_sensitivity(uid)
      if (!is.null(dps$dynamic_price_sensitivity)) dps$dynamic_price_sensitivity else NA
    })
  }
  # Add conversion probability if offer_id is present
  if ("user_id" %in% colnames(df) && "offer_id" %in% colnames(df)) {
    df$conversion_probability <- mapply(fetch_conversion_probability, df$user_id, df$offer_id)
  }
  # Add utility, CTR, trust columns if present in backend output
  if (!"utility_score" %in% colnames(df)) df$utility_score <- NA
  if (!"price_competitiveness_ctr" %in% colnames(df)) df$price_competitiveness_ctr <- NA
  if (!"trust_score" %in% colnames(df)) df$trust_score <- NA
  DT::datatable(
    df,
    options = list(
      scrollX = TRUE,
      scrollY = "400px",
      pageLength = 25,
      lengthMenu = c(10, 25, 50, 100),
      autoWidth = FALSE,
      columnDefs = list(list(targets = "_all", className = "dt-center"))
    ),
    rownames = FALSE,
    filter = "top",
    selection = "single"
  )
})

# --- Add/Update plot for market demand index by location ---
output$market_demand_index_plot <- renderPlotly({
  df <- values$user_market_table_new
  if (is.null(df) || nrow(df) == 0) return(NULL)
  p <- ggplot(df, aes(x = destination, y = demand_index, fill = market_state_label)) +
    geom_bar(stat = "identity") +
    labs(title = "Market Demand Index by Location", x = "Location", y = "Demand Index") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggplotly(p)
})

# --- Add tooltips/info boxes for new metrics ---
output$dynamic_price_sensitivity_info <- renderUI({
  HTML("<b>Dynamic Price Sensitivity:</b> Combines user base sensitivity, days to go, and price volatility in the destination. See README for formula.")
})
output$market_demand_index_info <- renderUI({
  HTML("<b>Market Demand Index:</b> Composite index of normalized price, booking urgency, price volatility, and competition density. See README for formula.")
})

# --- Add a new tab or box to show merged generated data ---
# Helper to load and merge the three CSVs
load_generated_data <- function(selected_user_ids) {
  # Read CSVs with explicit column types
  market_state <- tryCatch(read.csv("/data/market_state_by_location.csv", stringsAsFactors = FALSE), error = function(e) NULL)
  bandit_results <- tryCatch(read.csv("/data/bandit_simulation_results.csv", 
                                      colClasses = c(user_id = "character", offer_id = "character"), stringsAsFactors = FALSE), error = function(e) NULL)
  conversion_probs <- tryCatch(read.csv("/data/conversion_probabilities.csv", 
                                        colClasses = c(user_id = "character", offer_id = "character"), stringsAsFactors = FALSE), error = function(e) NULL)
  # Coerce keys to character
  if (!is.null(bandit_results)) {
    bandit_results$user_id <- as.character(bandit_results$user_id)
    bandit_results$offer_id <- as.character(bandit_results$offer_id)
  }
  if (!is.null(conversion_probs)) {
    conversion_probs$user_id <- as.character(conversion_probs$user_id)
    conversion_probs$offer_id <- as.character(conversion_probs$offer_id)
  }
  # Filter by selected user IDs if provided
  if (!is.null(selected_user_ids) && length(selected_user_ids) > 0) {
    bandit_results <- bandit_results[bandit_results$user_id %in% selected_user_ids, ]
    conversion_probs <- conversion_probs[conversion_probs$user_id %in% selected_user_ids, ]
  }
  # Merge bandit_results and conversion_probs on user_id, offer_id
  merged <- tryCatch({
    merge(bandit_results, conversion_probs, by = c("user_id", "offer_id"), all.x = TRUE)
  }, error = function(e) NULL)
  # Merge with market_state on destination/location
  if (!is.null(merged) && !is.null(market_state) && "destination" %in% colnames(merged)) {
    merged <- tryCatch({
      merge(merged, market_state, by.x = "destination", by.y = "location", all.x = TRUE)
    }, error = function(e) merged)
  }
  return(merged)
}

# Add a new output for the merged table
output$generated_data_table <- DT::renderDataTable({
  user_ids <- input$selected_users_for_offers
  data <- tryCatch({
    if (!file.exists("/data/market_state_by_location.csv") ||
        !file.exists("/data/bandit_simulation_results.csv") ||
        !file.exists("/data/conversion_probabilities.csv")) {
      showNotification("One or more required data files are missing.", type = "error")
      return(NULL)
    }
    load_generated_data(user_ids)
  }, error = function(e) {
    showNotification("Error loading generated data table.", type = "error")
    return(NULL)
  })
  if (is.null(data) || nrow(data) == 0) {
    showNotification("No generated data available for the selected user(s).", type = "warning")
    return(NULL)
  }
  DT::datatable(data, options = list(pageLength = 20, scrollX = TRUE), rownames = FALSE)
})

# Add a new tab or box in the UI for the generated data table
# For example, add to the 'Results & Analysis' tab or as a new tab
# Example UI addition:
# tabItem(tabName = "generated_data",
#   box(title = "Generated Data Overview", status = "primary", solidHeader = TRUE, width = 12,
#     DT::dataTableOutput("generated_data_table")
#   )
# )

}

# Run the application
print("=== STARTING SHINY APP ===")
write("Starting Shiny app", "/tmp/shiny_debug.log", append = TRUE)
shinyApp(ui = ui, server = server)

