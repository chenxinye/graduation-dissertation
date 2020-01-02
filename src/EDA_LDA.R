#@author:chenxinye
#@2019.06.09

library(tcltk)
library(tidytext)
library(magrittr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
library(reshape2)
library(wordcloud)
library(igraph)
library(ggraph)
library(widyr)
library(arules)
library(rattle)
library(arulesViz)
library(tm)
library(NLP)
library(tm) 
library(topicmodels)

set.seed(200)
data(stop_words)
setwd("D:/graduation project/src/data")
Combined_News_DJIA <- read.csv("Combined_News_DJIA.csv")

combine_topic <- function(Combined_News_DJIA,n){
  columns = names(Combined_News_DJIA)
  list_f = c()
  for (i in 1:length(Combined_News_DJIA[,1])){
    strl = ""
    for (col in columns[2:n]){
      strl = paste(strl,Combined_News_DJIA[col][i,], sep = "", collapse = NULL)
    }
    list_f = append(list_f,strl)}
  Combined_News_DJIA$combine_topic = list_f
  return(Combined_News_DJIA)}

tokens <- function(dfcol){
  #one token per document per row
  df_ = data.frame(line = 1:length(dfcol),text = dfcol)
  df_$text = as.character(df_$text)
  df_$text <- gsub("b'", "", df_$text)
  df_$text <- gsub("[0-9]", "", df_$text)
  df_ = df_%>% unnest_tokens(word, text)
  return(df_)}

DF = combine_topic(Combined_News_DJIA,26)
DF = DF$combine_topic %>% tokens() %>% anti_join(stop_words)
head(DF)

corpus <- Corpus(VectorSource(DF$word))
DTcorpus <- DocumentTermMatrix(corpus,control = list(wordLengths = c(1, Inf),bounds = list(global = 5, Inf),removeNumbers = TRUE))

meancosine_caculate <- function(Documentmatrix){
  mean_similarity <- c();mean_similarity[1] = 1
  for(i in 2:20){
    control <- list(burnin = 500, iter = 3000, keep = 100)
    Gibbs <- LDA(Documentmatrix, k = i, method = "Gibbs", control = control)
    term <- terms(Gibbs, 50) ;word <- as.vector(term)  ;freq <- table(word) ;unique_word <- names(freq)
    mat <- matrix(rep(0, i * length(unique_word)),nrow = i, ncol = length(unique_word))
    colnames(mat) <- unique_word
    for(k in 1:i){
      for(t in 1:50){mat[k, grep(term[t,k], unique_word)] <- mat[k, grep(term[t, k], unique_word)] + 1}}
    p <- combn(c(1:i), 2);l <- ncol(p);top_similarity <- c()
    for(j in 1:l){
      x <- mat[p[, j][1], ];y <- mat[p[, j][2], ]
      top_similarity[j] <- sum(x * y) / sqrt(sum(x^2) * sum(y ^ 2))}
    mean_similarity[i] <- sum(top_similarity) / l;message("top_num ", i)}
  return(mean_similarity)}

#cosine.similarity <- meancosine_caculate(DTcorpus)

#par(mfrow = c(1, 1))
#plot(cosine.similarity, type = "l")

gibbs <- LDA(DTcorpus, k = 6, method = "Gibbs", control = list(burnin = 500, iter = 1000, keep = 100))

termsl <- terms(gibbs, 50)

write.csv(termsl, "count/ldaterms.csv", row.names = FALSE)

plot_LDA <- function(terms,data = pdata.freq){
  len1 <- length(terms[,1]);len2 <- length(terms[1,])
  vec <- vector(mode = "logical",length = len2);vec[1] = 0.0
  for (i in 1:len2){count <- 0.0
  for (j in 1:len1){
    freq <- data[which(data$Var1 == terms[j,i]),c("Freq")]
    count <- count + freq}
  count %>% print();vec[i] <- count}
  dataf <- data.frame(vec = vec,topic = as.character(1:len2))
  myLabel = as.vector(dataf$topic)
  myLabel = paste("topic:",myLabel, "(", round(dataf$vec / sum(dataf$vec) * 100, 2), "%)", sep = "")
  
  p = ggplot(dataf, aes(x = "", y = vec, fill = factor(topic))) + 
    geom_bar(stat = "identity", width = 1) +    
    coord_polar(theta = "y") + 
    labs(x = "", y = "", title = "") + 
    theme(axis.ticks = element_blank()) + 
    theme(legend.title = element_blank(), legend.position = "top") + 
    scale_fill_discrete(breaks = dataf$topic, labels = myLabel)
  print(p)}

picture_output <- function(pos_cos) {
  cosdf <- data.frame(x = 1:length(pos_cos),meancosine = pos_cos,x = rep("topic",10))
  p <- ggplot(cosdf,aes(x= x,y= meancosine)) 
  p <- p + stat_smooth(se = TRUE) + geom_point();print(p)}

freq <- DF %>% count(word,sort = T)
freq$n <- freq$n/sum(freq$n)
names(freq) = c("Var1","Freq")

plot_LDA(termsl,freq)
#picture_output(cosine.similarity)