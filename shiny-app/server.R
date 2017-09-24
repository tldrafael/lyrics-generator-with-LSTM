require(shiny)
library(keras)

#------- LOAD AND SET MAIN VARIABLES -----------------------------------------------------

model <- load_model_hdf5("../models/last-release.h5")

words_encoding <- readRDS("../helpers/words-enconding.rds")
words_selected <- readRDS("../helpers/words-lexical.rds")

timesteps <- 3


#------- UTILS FUNCTIONS -----------------------------------------------------------------

one_hot_encoding <- function(word){
        as.character(word) %>% 
                words_encoding[.] %>% 
                unname() %>% 
                unlist()
}



get_best_indices <- function(preds, n){
        preds_order <- sort(preds, TRUE)
        preds_order_best <- preds_order[1:n]
        preds_best_indices <- which(preds%in%preds_order_best)
        
        return(preds_best_indices)
}


predict_lyrics <- function(n_diversity, n_diversity_min){
        
        lyrics <- "start_lyrics"
        x_sample <- rep("start_lyrics", timesteps)
        
        for( i in 1:100){
                # define the range of choice from prediction
                n_diversity_adj <- (n_diversity - n_diversity_min)*exp(-2*i/10) + n_diversity_min
                
                # transfrom the sample of 5 words in hot enconded
                x_sample_encoded <- sapply(x_sample, one_hot_encoding)
                x_sample_encoded <- t(x_sample_encoded)
                x_sample_encoded <- array(x_sample_encoded, dim=c(1, dim(x_sample_encoded)))
                
                # predict the most probable words
                y_sample_preds <- predict(model, x_sample_encoded)
                y_sample_preds_best <- get_best_indices(y_sample_preds, n=n_diversity_adj)
                
                # adjust vector if length equals 1 to avoid bug that happens with sample function
                if( length(y_sample_preds_best)==1){
                        y_sample_preds_best <- rep(y_sample_preds_best, 2)
                }
                
                y_sample_pred_chosen <- sample(y_sample_preds_best, 1)
                y_sample_pred_word <- words_selected[y_sample_pred_chosen]
                
                # join new word to lyrics and to next feature sample
                lyrics <- c(lyrics, y_sample_pred_word)
                x_sample <- c(x_sample[-1], y_sample_pred_word)
                
                # check if is predict the end of the lyrics
                if( y_sample_pred_word == "end_lyrics"){
                        return(lyrics[-length(lyrics)])
                }
                
        }
        
        return(lyrics)
}


print_format_lyrics <- function(xlyrics){
        xlyrics[ ! xlyrics %in% c("start_lyrics")] %>% 
                paste0(collapse=" ") %>% 
                strsplit("end_paragraph") %>% 
                unlist()
}

updateLyrics <- function(diversity_max=5, diversity_min=3){
        lyrics <- predict_lyrics(diversity_max, diversity_min)        
        lyrics <- print_format_lyrics(lyrics)
        return(lyrics)
}



#------- SHINY SERVER --------------------------------------------------------------------

shinyServer(function(input, output, session){
        
        observeEvent(input$create_lyrics,
                     output$lyrics_new <- renderText({ 
                                updateLyrics(as.numeric(input$diversity_max), 
                                             as.numeric(input$diversity_min ))
                             })
                     )
        
        
})