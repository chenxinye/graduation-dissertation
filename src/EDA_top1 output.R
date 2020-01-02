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

tokens <- function(dfcol){
  #one token per document per row
  df_ = data.frame(line = 1:length(dfcol),text = dfcol)
  df_$text = as.character(df_$text)
  df_$text <- gsub("b'", "", df_$text)
  df_$text <- gsub("[0-9]", "", df_$text)
  df_ = df_%>% unnest_tokens(word, text)
  return(df_)
}

combine <- function(df1,df2,str1,str2){
  m_1 = mutate(df1,topic = str1)
  m_2 = mutate(df2,topic = str2)
  freq <- bind_rows(m_1,m_2)
  return(freq)
}
emotion_w <- get_sentiments("bing")


data1.2 <- Combined_News_DJIA$Top1%>% tokens() %>% anti_join(stop_words)  %>% left_join(emotion_w) 
data <- data1.2%>% count(index = line,sentiment) %>% spread(sentiment,n,fill = 0) %>% mutate(sentiment = positive - negative)
write.csv(data,file = 'emotion_ev/emotion_evaluation.csv',row.names = FALSE)
