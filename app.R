library(shiny)
library(readr)
library(readxl)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)

ui <- fluidPage(
  titlePanel("Application interactive de Data Analysis"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file","CSV ou Excel", accept=c(".csv",".xlsx")),
      textInput("sep","Séparateur CSV", value=","),
      textInput("sheet","Nom de la feuille (Excel)", value=""),
      hr(),
      uiOutput("target_ui"),
      uiOutput("features_ui"),
      selectInput("model","Modèle", choices=c("Régression linéaire","Random Forest"), selected="Random Forest"),
      sliderInput("split","Taille test (%)", min=10, max=50, value=20)
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Aperçu", tableOutput("head"), tableOutput("num_desc"), tableOutput("cat_desc")),
        tabPanel("Univarié", uiOutput("uni_ui"), plotOutput("uni_plot"), plotOutput("uni_box")),
        tabPanel("Bivarié", uiOutput("bi_ui"), plotOutput("bi_plot")),
        tabPanel("Multivarié", plotOutput("corr_plot")),
        tabPanel("Modélisation", verbatimTextOutput("task"), verbatimTextOutput("metrics"))
      )
    )
  )
)

server <- function(input, output, session){
  
  data <- reactive({
    req(input$file)
    name <- tolower(input$file$name)
    if(grepl("\\.csv$", name)){
      read_csv(input$file$datapath, show_col_types = FALSE, progress = FALSE, col_types = cols(), delim=input$sep)
    } else {
      read_excel(input$file$datapath, sheet = ifelse(nchar(input$sheet)>0, input$sheet, 1))
    }
  })
  
  observeEvent(data(), {
    updateSelectInput(session, "model", choices=c("Régression linéaire","Random Forest"))
  })
  
  output$head <- renderTable({
    head(data(), 10)
  })
  
  # Colonnes
  num_cols <- reactive({
    colnames(dplyr::select_if(data(), is.numeric))
  })
  cat_cols <- reactive({
    setdiff(colnames(data()), num_cols())
  })
  
  output$num_desc <- renderTable({
    if(length(num_cols())==0) return(NULL)
    as.data.frame(summary(data()[, num_cols(), drop=FALSE]))
  })
  output$cat_desc <- renderTable({
    if(length(cat_cols())==0) return(NULL)
    apply(data()[, cat_cols(), drop=FALSE], 2, function(x) head(sort(table(as.character(x)), decreasing=TRUE), 10)) %>% as.data.frame()
  })
  
  output$uni_ui <- renderUI({
    selectInput("uni_col","Variable", choices=colnames(data()))
  })
  output$bi_ui <- renderUI({
    fluidRow(
      column(4, selectInput("x","X", choices=colnames(data()))),
      column(4, selectInput("y","Y", choices=colnames(data()))),
      column(4, selectInput("hue","Couleur (optionnel)", choices=c("(aucune)", colnames(data()))))
    )
  })
  
  output$target_ui <- renderUI({
    req(data())
    selectInput("target","Variable cible (y)", choices=colnames(data()))
  })
  output$features_ui <- renderUI({
    req(data(), input$target)
    selectizeInput("features","Variables explicatives (X)",
                   choices=setdiff(colnames(data()), input$target),
                   multiple=TRUE, selected=head(setdiff(colnames(data()), input$target), 5))
  })
  
  output$uni_plot <- renderPlot({
    req(input$uni_col)
    df <- data()
    col <- input$uni_col
    if(col %in% num_cols()){
      ggplot(df, aes(x=.data[[col]])) + geom_histogram(bins=30) + ggtitle(paste("Histogramme de", col))
    } else {
      ggplot(df, aes(x=.data[[col]])) + geom_bar() + coord_flip() + ggtitle(paste("Barplot de", col))
    }
  })
  
  output$uni_box <- renderPlot({
    req(input$uni_col)
    df <- data(); col <- input$uni_col
    if(col %in% num_cols()){
      ggplot(df, aes(y=.data[[col]])) + geom_boxplot() + ggtitle(paste("Boxplot de", col))
    }
  })
  
  output$bi_plot <- renderPlot({
    req(input$x, input$y)
    df <- data(); x <- input$x; y <- input$y; h <- if(input$hue=="(aucune)") NULL else input$hue
    if(x %in% num_cols() && y %in% num_cols()){
      ggplot(df, aes(x=.data[[x]], y=.data[[y]], color=if(is.null(h)) NULL else .data[[h]])) + geom_point()
    } else if(x %in% cat_cols() && y %in% num_cols()){
      ggplot(df, aes(x=.data[[x]], y=.data[[y]])) + geom_boxplot()
    } else if(x %in% num_cols() && y %in% cat_cols()){
      ggplot(df, aes(x=.data[[y]], y=.data[[x]])) + geom_boxplot()
    } else {
      tab <- table(as.character(df[[x]]), as.character(df[[y]]))
      df_tab <- as.data.frame(tab)
      ggplot(df_tab, aes(Var1, Var2, fill=Freq)) + geom_tile()
    }
  })
  
  output$corr_plot <- renderPlot({
    if(length(num_cols())<2) return(NULL)
    c <- cor(data()[, num_cols(), drop=FALSE], use="pairwise.complete.obs")
    dfc <- as.data.frame(as.table(c))
    ggplot(dfc, aes(Var1, Var2, fill=Freq)) + geom_tile() + ggtitle("Heatmap des corrélations")
  })
  
  # Modélisation
  output$task <- renderText({
    req(input$target, input$features)
    y <- data()[[input$target]]
    is_classif <- !is.numeric(y) || length(unique(y)) <= max(10, floor(0.05*nrow(data())))
    paste0("Tâche détectée : ", if(is_classif) "Classification" else "Régression")
  })
  
  output$metrics <- renderText({
    req(input$target, input$features)
    df <- data()
    df <- df[complete.cases(df[, c(input$target, input$features), drop=FALSE]), ]
    y <- df[[input$target]]
    X <- df[, input$features, drop=FALSE]
    
    is_classif <- !is.numeric(y) || length(unique(y)) <= max(10, floor(0.05*nrow(df)))
    
    set.seed(42)
    train_idx <- createDataPartition(y, p = 1 - input$split/100, list = FALSE)
    X_train <- X[train_idx, , drop=FALSE]; y_train <- y[train_idx]
    X_test  <- X[-train_idx, , drop=FALSE]; y_test  <- y[-train_idx]
    
    # Encodage simple des catégorielles
    for(cn in colnames(X_train)){
      if(!is.numeric(X_train[[cn]])){
        X_train[[cn]] <- as.factor(X_train[[cn]])
        X_test[[cn]]  <- factor(X_test[[cn]], levels=levels(X_train[[cn]]))
      }
    }
    
    if(is_classif){
      if(input$model == "Random Forest"){
        fit <- randomForest(x=X_train, y=as.factor(y_train))
        pred <- predict(fit, X_test)
      } else {
        # Logistic sur données one-vs-rest si binaire
        if(length(unique(y_train)) == 2){
          fit <- train(x=X_train, y=as.factor(y_train), method="glm", family="binomial")
          pred <- predict(fit, X_test)
        } else {
          fit <- randomForest(x=X_train, y=as.factor(y_train)) # fallback multi-classes
          pred <- predict(fit, X_test)
        }
      }
      acc <- caret::confusionMatrix(as.factor(pred), as.factor(y_test))$overall["Accuracy"]
      paste0("Accuracy: ", round(acc,3))
    } else {
      if(input$model == "Random Forest"){
        fit <- randomForest(x=X_train, y=y_train)
        pred <- predict(fit, X_test)
      } else {
        fit <- train(x=X_train, y=y_train, method="lm")
        pred <- predict(fit, X_test)
      }
      r2 <- caret::R2(pred, y_test)
      rmse <- caret::RMSE(pred, y_test)
      mae <- mean(abs(pred - y_test))
      paste0("R²: ", round(r2,3), " | RMSE: ", round(rmse,3), " | MAE: ", round(mae,3))
    }
  })
}

shinyApp(ui, server)
