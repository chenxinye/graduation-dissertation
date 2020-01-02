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

data(stop_words)
setwd("D:/graduation project/src/data")
Combined_News_DJIA <- read.csv("Combined_News_DJIA.csv")
head(Combined_News_DJIA)

combine_topic <- function(Combined_News_DJIA,n){
  columns = names(Combined_News_DJIA)
  list_f = c()
  for (i in 1:length(Combined_News_DJIA[,1])){
    strl = ""
    for (col in columns[2:n]){
      strl = paste(strl,Combined_News_DJIA[col][i,], sep = "", collapse = NULL)}
    list_f = append(list_f,strl)
  }
  Combined_News_DJIA$combine_topic = list_f
  return(Combined_News_DJIA)
}

df <- combine_topic(Combined_News_DJIA,26) #topic 2
df <- df[c("combine_topic","Date","Label")]
df$combine_topic <- gsub("b'", "", df$combine_topic)
df$combine_topic <- gsub("[0-9]", "", df$combine_topic)
df$combine_topic <- gsub("\\n", "", df$combine_topic)
df$combine_topic <- gsub("\\\\", "", df$combine_topic)
df$combine_topic <- gsub("\"", "", df$combine_topic)

write.csv(df,file = "text/df.csv",row.names = FALSE)

dfword <- df$combine_topic%>% tokens() %>% anti_join(stop_words)
emotion_w <- get_sentiments("bing")
names(dfword) <- c("line","word")
data1.2 <- dfword %>% inner_join(emotion_w) 
#data1.2 <- data1.2%>% count(topic,index = line,sentiment) %>% spread(sentiment,n,fill = 0) %>% mutate(sentiment = 2*positive - negative)
#write.csv(data1.2,file = 'count/emotion_eval.csv')
data <- data1.2%>% count(line,sentiment) %>% spread(sentiment,n,fill = 0) %>% mutate(sentiment = 3*positive - negative)
write.csv(data,file = "text/emotion_value.csv",row.names = FALSE)