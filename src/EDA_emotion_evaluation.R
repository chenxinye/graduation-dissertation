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

data(stop_words)
setwd("D:/graduation project/src/data")
Combined_News_DJIA <- read.csv("Combined_News_DJIA.csv")
head(Combined_News_DJIA)

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

df <- Combined_News_DJIA
dt1 = df$Top1 %>% tokens() %>% anti_join(stop_words)
dt2 = df$Top2 %>% tokens() %>% anti_join(stop_words)
dt3 = df$Top3 %>% tokens() %>% anti_join(stop_words)
dt4 = df$Top4 %>% tokens() %>% anti_join(stop_words)
dt5 = df$Top5 %>% tokens() %>% anti_join(stop_words)
dt6 = df$Top6 %>% tokens() %>% anti_join(stop_words)
dt7 = df$Top7 %>% tokens() %>% anti_join(stop_words)
dt8 = df$Top8 %>% tokens() %>% anti_join(stop_words)
dt9 = df$Top9 %>% tokens() %>% anti_join(stop_words)
dt10 = df$Top10 %>% tokens() %>% anti_join(stop_words)

emotion_w <- get_sentiments("bing")
df_1 <- combine(dt1,dt2,'topic 1','topic 2')
df_2 <- combine(dt3,dt4,'topic 3','topic 4')
df_3 <- combine(dt5,dt6,'topic 5','topic 6')
df_4 <- combine(dt7,dt8,'topic 7','topic 8')
df_5 <- combine(dt9,dt10,'topic 9','topic 10')

df = bind_rows(df_1,df_2)
df = bind_rows(df,df_3)
df = bind_rows(df,df_4)
df = bind_rows(df,df_5)

data1.2 <- df %>% inner_join(emotion_w) 
#data1.2 <- data1.2%>% count(topic,index = line,sentiment) %>% spread(sentiment,n,fill = 0) %>% mutate(sentiment = 2*positive - negative)
#write.csv(data1.2,file = 'count/emotion_eval.csv')
data1.2 <- data1.2%>% count(topic,index = line,sentiment) %>% spread(sentiment,n,fill = 0) %>% mutate(sentiment = 2*positive - negative)
p <- ggplot(data1.2,aes(index,sentiment,fill = topic)) + geom_col(show.legend = F) + facet_wrap(~topic,ncol = 2, scales = 'free_x')
print(p)

write.csv(data1.2,file = 'count/emotion_eval.csv')
dg = get_sentiments("bing") %>% count(sentiment)
print(paste("negative vs positive:",dg$n[1] /  dg$n[2]))

combine_topic <- function(Combined_News_DJIA,n){
  columns = names(Combined_News_DJIA)
  list_f = c()
  for (i in 1:length(Combined_News_DJIA[,1])){
    strl = ""
    for (col in columns[2:n]){
      strl = paste(strl,Combined_News_DJIA[col][i,], sep = "", collapse = NULL)
    }
    list_f = append(list_f,strl)
  }
  Combined_News_DJIA$combine_topic = list_f
  return(Combined_News_DJIA)
}

df_all <- combine_topic(Combined_News_DJIA,26)
df_all <- df_all$combine_topic %>% tokens() %>% anti_join(stop_words)
word_em_count <- df_all %>% inner_join(get_sentiments("bing")) %>% count(word,sentiment,sort = T) %>% ungroup()

word_em_count %>% group_by(sentiment) %>% top_n(20) %>%
  ungroup() %>% mutate(word = reorder(word,n,fill=sentiment))%>%
  ggplot(aes(word,n,fill = sentiment)) + geom_col(show.legend = F) + facet_wrap(~sentiment,scales = 'free_y') + 
    labs(y = "contribution",x = NULL) + coord_flip()

df_all %>% inner_join(get_sentiments("bing")) %>% count(word,sentiment,sort = T) %>% 
  acast(word~sentiment, value.var = "n",fill = 0) %>%
  comparison.cloud(colors=c("DarkRed","#FFA500"),max.words = 120,title.size = 1.5)

help(comparison.cloud)
