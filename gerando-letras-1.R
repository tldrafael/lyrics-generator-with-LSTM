library(keras)
library(tidyverse)
library(stringr)
library(tokenizers)


#--------------------------------------------------------------------------------------
# Crawl data
#--------------------------------------------------------------------------------------

link_base <- 'https://www.letras.mus.br'

# listando os links
ws_links <- paste0(link_base, '/wesley-safadao/') %>% 
        rvest::html_session() %>% 
        rvest::html_nodes('.cnt-list--alp > ul > li > a') %>% 
        rvest::html_attr('href')


limpar_musica <- function(txt){
        txt %>% 
                stringr::str_trim() %>% 
                stringr::str_to_lower() %>% 
                stringr::str_replace_all('[^a-z0-9êâôáéíóúãõàç;,!?: \n-]', '') %>% 
                stringr::str_replace_all('([ ,?!])+', '\\1') %>% 
                stringr::str_replace_all(' ([;,!?:-])', '\\1') %>% 
                stringr::str_replace_all('\n{3,}', '\n\n')
}


pegar_letra <- function(link){
        # do link  até a parte que tem o conteúdo
        result <- paste0(link_base, link) %>% 
                rvest::html_session() %>% 
                rvest::html_nodes('.cnt-letra > article > p') %>% 
                # Pegue o texto com as tags html para pegar os \n
                as.character() %>% 
                stringr::str_replace_all('<[brp/]+>', '\n') %>% 
                paste(collapse='\n\n') %>% 
                # Limpeza do texto
                limpar_musica() %>% 
                tokenizers::tokenize_characters(strip_non_alphanum=FALSE, simplify=TRUE)
        c(result, '@')
        
}


pega_letra_por_palavras <- function(link){
        result <- paste0(link_base, link) %>% 
                rvest::html_session() %>% 
                rvest::html_nodes('.cnt-letra > article > p') %>% 
                # Pegue o texto com as tags html para pegar os \n
                as.character() %>% 
                stringr::str_replace_all('<[brp/]+>', '\n') %>% 
                paste(collapse='\n\n') %>% 
                # Limpeza do texto
                limpar_musica() %>% 
                str_replace_all("\n", " end_paragraph ") %>% 
                tokenize_words(simplify=TRUE) 
        c('start_lyrics', result, 'end_lyrics')        
}


p <- progress::progress_bar$new(total=length(ws_links))
ws_letras <- unlist(purrr::map(ws_links, ~{
        p$tick()
        pega_letra_por_palavras(.x)
}))

cat(head(ws_letras, which(ws_letras == 'end_lyrics')[1] - 1), sep = ' ')



#--------------------------------------------------------------------------------------
# Filtra dados, seleciona somente pavras mais relevantes
# Estrutura dados com relação temporal
#--------------------------------------------------------------------------------------

data <- tibble(X = ws_letras) %>% 
                mutate(Y = c(lead(X, 1, default=tail(X, 1)))) 

words_selected <- data %>% 
                        group_by(X) %>% 
                        summarise(count=n()) %>% 
                        filter(count>50) %>% 
                        select(X) %>% 
                        unlist()


# preparing the words sequence
data_selected <- data %>% 
                        #filter(X %in% words_selected & X!="end_paragraph") %>%
                        filter(X %in% words_selected) %>% 
                        mutate(X1 = lag(X, 1, default=X[1]))  %>% 
                        mutate(X2 = lag(X1, 1, default=X1[1])) %>% 
                        mutate(X3 = lag(X2, 1, default=X2[1])) %>% 
                        mutate(X4 = lag(X3, 1, default=X3[1])) %>% 
                        mutate(X5 = lag(X4, 1, default=X4[1])) %>% 
                        mutate(Y = c(lead(X, 1, default=tail(X, 1)))) %>% 
                        select(X5, X4, X3, X2, X1, X, Y) 

# reduce number of end_paragraph word output
end_paragraph_idx <- data_selected$Y %>%  unlist() %>% {.=="end_paragraph"} %>% which()
end_paragraph_idx_not <- data_selected$Y %>%  unlist() %>% {.!="end_paragraph"} %>% which()
words_idx_selected <- c(end_paragraph_idx_not,
                        sample(end_paragraph_idx, 1500)
                        )
data_selected <- data_selected %>% slice(words_idx_selected)


# features_dim
## words - one-hot-encoding
words_n <- length(words_selected)

encoding <-rep(0, words_n)
words_encoding <- list()
for( w in words_selected){
        aux <- encoding
        aux[which(words_selected==w)] <- 1
        words_encoding[[w]] <- aux
}

#data_selected_X <- data_selected %>% select(X5, X4, X3, X2, X1, X)
timesteps <- 4
data_selected_X <- data_selected %>% select(paste0("X", timesteps:1))
data_selected_Y <- data_selected %>% select(Y)


#--------------------------------------------------------------------------------------
# Utils functions
#--------------------------------------------------------------------------------------

one_hot_encoding <- function(w){
        as.character(w) %>% 
                words_encoding[.] %>% 
                unname() %>% 
                unlist()
}


capture_encode <- function(rowX, colX){
        colX_values <- data_selected_X[rowX, colX]
        one_hot_encoding(colX_values)
}


get_sample <- function(rowX, colX=1:timesteps){
        X_train <- lapply(X=colX, capture_encode, rowX=rowX) %>% 
                        lapply(as.numeric) %>% 
                        as.data.frame() %>% unname() %>% 
                        as.matrix()        
        
        X_train <- t(X_train)
        X_train <- array(X_train, dim=c(1, dim(X_train)))
        
        Y_train <- one_hot_encoding(data_selected_Y$Y[rowX])
        Y_train <- t(Y_train)
        
        res <- list(X_train=X_train, Y_train=Y_train)
        return( res)
}



#--------------------------------------------------------------------------------------
# Transform the whole dataset in one-hot-enconding format
#--------------------------------------------------------------------------------------

train_onehot_X <- array(data=NA, dim=c(nrow(data_selected), timesteps, words_n))
train_onehot_Y <- array(data=NA, dim=c(nrow(data_selected), words_n))
for( i in seq(nrow(data_selected_X))){
        s <- get_sample(i)
        train_onehot_X[i, , ] <- s$X_train
        train_onehot_Y[i, ] <- s$Y_train        
}


#--------------------------------------------------------------------------------------
# Training      
#--------------------------------------------------------------------------------------

units_n <- 32*4
batch_size <- 66504
model <- keras_model_sequential() %>% 
                layer_lstm(units_n, batch_input_shape=c(batch_size, timesteps, words_n)) %>% 
                layer_dense(words_n) %>% 
                layer_activation("softmax")

model %>% 
        compile( loss = "categorical_crossentropy",
                 optimizer = optimizer_adam(lr=1e-4) )

# validation_split=0.112478
idx_until <- (nrow(train_onehot_X) %/% batch_size) * batch_size
model %>%    
        fit(x=train_onehot_X[1:idx_until,,], y=train_onehot_Y[1:idx_until,], 
                epochs=50, stateful=TRUE, shuffle=TRUE, 
                batch_size=batch_size, validation_split=0) %>% 
                {.} -> model_loss_hist

plot(model_loss_hist$metrics$loss, col="red", type="l", ylim=c(0,6))
lines(model_loss_hist$metrics$val_loss, col="green", type="l")

model_new <- keras_model_sequential() %>% 
                layer_lstm(units_n, batch_input_shape=c(1, timesteps, words_n)) %>% 
                layer_dense(words_n) %>% 
                layer_activation("softmax")

model_new$set_weights(get_weights(model)) 


save_model_hdf5(model_new, "model-250-epochs.rds")

X_subset_idx <- sample(nrow(data_selected_X), 1e4)
X_subset_idx <- seq(nrow(data_selected_X))
X_subset <- data_selected_X[X_subset_idx, ]
X_subset$Y <- data_selected_Y$Y[X_subset_idx]
X_subset$H <- ""
for( i in seq(nrow(X_subset))){
        test_sample <- get_sample(i)$X_train
        preds <- predict(model_new, test_sample)
        preds_idx <- which.max(preds)
        X_subset$H[i] <- words_selected[preds_idx]
}
X_subset$H %>% table() %>% sort(T)
View(X_subset)


#--------------------------------------------------------------------------------------
# Gera letras
#--------------------------------------------------------------------------------------

get_best_indices <- function(preds, n){
        preds_order <- sort(preds, TRUE)
        preds_order_best <- preds_order[1:n]
        preds_best_indices <- which(preds%in%preds_order_best)
        
        return(preds_best_indices)
}


generate_lyrics <- function(n_diversity=5, n_diversity_min=3){
        lyrics <- "start_lyrics"
        x_sample <- rep("start_lyrics", timesteps)
        
        for( i in seq(100)){
                
                n_diversity_adj <- (n_diversity - n_diversity_min)*exp(-2*i/10) + n_diversity_min
                
                x_sample_encoded <- sapply(x_sample, one_hot_encoding)
                y_sample_preds <- predict(model_new, test_sample)
                
                y_samples_best_preds <- get_best_indices(y_sample_preds, n=n_diversity_adj)
                # just to adjust function sample, which does not recognize a vector of one dim
                if( length(y_samples_best_preds)==1 ){
                        y_samples_best_preds <- rep(y_samples_best_preds, 2)
                }
                
                y_samples_chosen <- sample(y_samples_best_preds, 1)
                y_sample_word <- words_selected[y_samples_chosen]
                
                lyrics <- c(lyrics, y_sample_word)
                x_sample <- c(x_sample[-1], y_sample_word)
                
                if( y_sample_word == "end_lyrics"){
                        return(lyrics)
                }
                
        }
        
        return(lyrics)
}

print_lyrics <- function(xlyrics){
        xlyrics[ ! xlyrics %in% c("start_lyrics")] %>% 
                paste0(collapse=" ") %>% 
                strsplit("end_paragraph") %>% 
                unlist() %>% 
                return()
}


generate_lyrics(100, 10) %>% 
        print_lyrics()


