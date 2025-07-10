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
      menuItem("Market Analysis", tabName = "market", icon = icon("globe"))
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
          # Strategy Configuration
          box(
            title = "Strategy Configuration", status = "primary", solidHeader = TRUE, width = 4,
            
            h4("Market Conditions"),
            div(class = "metric-box",
                div(class = "metric-value", textOutput("days_until_travel")),
                div(class = "metric-label", "Days Until Travel")
            ),
            
            selectInput("market_demand_override", "Market Demand Override:",
                       choices = c("Auto-Detect" = "auto", "Low" = "Low", 
                                 "Medium" = "Medium", "High" = "High"),
                       selected = "auto"),
            
            sliderInput("price_sensitivity_override", "Price Sensitivity Override:",
                       min = 0, max = 1, value = 0.5, step = 0.1),
            
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
          ),
          
          # RL Recommendations
          box(
            title = "RL Model Recommendations", status = "info", solidHeader = TRUE, width = 8,
            
            h4("Market Analysis"),
            verbatimTextOutput("market_analysis"),
            
            h4("Strategy Recommendations"),
            div(id = "recommendations_display",
                div(class = "recommendation-box",
                    h5("Recommendation Engine"),
                    p("Apply a strategy to see AI-powered recommendations based on current market conditions and user behavior.")
                )
            ),
            
            h4("Customer Behavior Prediction"),
            fluidRow(
              column(3, div(class = "metric-box",
                           div(class = "metric-value", textOutput("predicted_conversion")),
                           div(class = "metric-label", "Conversion Rate"))),
              column(3, div(class = "metric-box",
                           div(class = "metric-value", textOutput("booking_urgency")),
                           div(class = "metric-label", "Booking Urgency"))),
              column(3, div(class = "metric-box",
                           div(class = "metric-value", textOutput("price_sensitivity_calc")),
                           div(class = "metric-label", "Price Sensitivity"))),
              column(3, div(class = "metric-box",
                           div(class = "metric-value", textOutput("brand_preference")),
                           div(class = "metric-label", "Top Brand")))
            )
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
    data_loaded = FALSE  # Track if data has been loaded
  )
  
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
        } else {
          values$user_offers <- NULL
          print("No data in response, set values$user_offers to NULL")
          write("No data in response, set values$user_offers to NULL", "/tmp/shiny_debug.log", append = TRUE)
          showNotification("No offers data found", type = "warning")
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

  # On app load, fetch the offers table
  fetch_and_update_offers()

  # Test button handler
  observeEvent(input$test_btn, {
    print("Test button clicked!")
    write("Test button clicked", "/tmp/shiny_debug.log", append = TRUE)
    showNotification("App is working! Test button clicked.", type = "default")
  })
  
  # Helper to fetch and update user IDs from /trial_user_ids
  fetch_and_update_user_ids <- function() {
    tryCatch({
      res <- GET(paste0(API_URL, "/trial_user_ids"))
      if (http_status(res)$category == "Success") {
        result <- fromJSON(content(res, "text", encoding = "UTF-8"))
        if (!is.null(result$user_ids) && length(result$user_ids) > 0) {
          values$user_ids <- result$user_ids
          updateSelectInput(session, "user_profile_select_offers", choices = result$user_ids)
        } else {
          values$user_ids <- NULL
          updateSelectInput(session, "user_profile_select_offers", choices = NULL)
        }
      } else {
        values$user_ids <- NULL
        updateSelectInput(session, "user_profile_select_offers", choices = NULL)
      }
    }, error = function(e) {
      values$user_ids <- NULL
      updateSelectInput(session, "user_profile_select_offers", choices = NULL)
    })
  }

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
    # After backend samples, reload offers and user IDs
    fetch_and_update_offers()
    fetch_and_update_user_ids()
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
    df
  })

  # After successful scenario save, fetch the offers table and user IDs
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
        
        # Fetch the updated offers table from backend to show the saved data
        fetch_and_update_offers()
        fetch_and_update_user_ids()
        
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
  })

  # --- TAB 2: STRATEGY SELECTION ---
  
  # Calculate days until travel
  output$days_until_travel <- renderText({
    if (is.null(values$scenario_data)) return("--")
    as.character(values$scenario_data$market_conditions$days_until_travel)
  })
  
  # Market analysis
  output$market_analysis <- renderText({
    if (is.null(values$market_conditions)) return("No market data available")
    
    mc <- values$market_conditions
    paste(
      paste("Destination:", mc$destination),
      paste("Market Demand:", mc$market_demand),
      paste("Demand Score:", round(mc$demand_score, 2)),
      paste("Seasonal Factor:", round(mc$seasonal_factor, 2)),
      paste("Price Volatility:", round(mc$price_volatility, 2)),
      paste("Booking Velocity:", round(mc$booking_velocity, 2)),
      sep = "\n"
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
  
  # Price trends plot (placeholder)
  output$price_trends_plot <- renderPlotly({
    # Generate sample price trend data
    dates <- seq(from = Sys.Date() - 30, to = Sys.Date(), by = "day")
    prices <- 150 + cumsum(rnorm(length(dates), 0, 5))
    
    plot_data <- data.frame(
      Date = dates,
      Average_Price = prices
    )
    
    p <- ggplot(plot_data, aes(x = Date, y = Average_Price)) +
      geom_line(color = "#3498db", size = 1.2) +
      geom_smooth(method = "loess", se = TRUE, alpha = 0.3) +
      labs(
        title = "30-Day Price Trend Analysis",
        x = "Date",
        y = "Average Price ($)"
      ) +
      theme_minimal()
    
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
}

# Run the application
print("=== STARTING SHINY APP ===")
write("Starting Shiny app", "/tmp/shiny_debug.log", append = TRUE)
shinyApp(ui = ui, server = server)

