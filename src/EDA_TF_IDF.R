#@author:chenxinye
#@2019.06.09

#TF-IDF
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
dt11 = df$Top11 %>% tokens() %>% anti_join(stop_words)
dt12 = df$Top12 %>% tokens() %>% anti_join(stop_words)
dt13 = df$Top13 %>% tokens() %>% anti_join(stop_words)
dt14 = df$Top14 %>% tokens() %>% anti_join(stop_words)
dt15 = df$Top15 %>% tokens() %>% anti_join(stop_words)
dt16 = df$Top16 %>% tokens() %>% anti_join(stop_words)
dt17 = df$Top17 %>% tokens() %>% anti_join(stop_words)
dt18 = df$Top18 %>% tokens() %>% anti_join(stop_words)
dt19 = df$Top19 %>% tokens() %>% anti_join(stop_words)
dt20 = df$Top20 %>% tokens() %>% anti_join(stop_words)
dt21 = df$Top21 %>% tokens() %>% anti_join(stop_words)
dt22 = df$Top22 %>% tokens() %>% anti_join(stop_words)
dt23 = df$Top23 %>% tokens() %>% anti_join(stop_words)
dt24 = df$Top24 %>% tokens() %>% anti_join(stop_words)
dt25 = df$Top25 %>% tokens() %>% anti_join(stop_words)

df_1 <- combine(dt1,dt2,'topic 1','topic 2')
df_2 <- combine(dt3,dt4,'topic 3','topic 4')
df_3 <- combine(dt5,dt6,'topic 5','topic 6')
df_4 <- combine(dt7,dt8,'topic 7','topic 8')
df_5 <- combine(dt9,dt10,'topic 9','topic 10')
df_6 <- combine(dt11,dt12,'topic 11','topic 12')
df_7 <- combine(dt13,dt14,'topic 13','topic 14')
df_8 <- combine(dt15,dt16,'topic 15','topic 16')
df_9 <- combine(dt17,dt18,'topic 17','topic 18')
df_10 <- combine(dt19,dt20,'topic 19','topic 20')
df_11 <- combine(dt21,dt22,'topic 21','topic 22')
df_12 <- combine(dt23,dt24,'topic 23','topic 24')
df_13 = mutate(dt25,topic = 'topic 25')

df = bind_rows(df_1,df_2)
df = bind_rows(df,df_3)
df = bind_rows(df,df_4)
df = bind_rows(df,df_5)
df = bind_rows(df,df_6)
df = bind_rows(df,df_7)
df = bind_rows(df,df_8)
df = bind_rows(df,df_9)
df = bind_rows(df,df_10)
df = bind_rows(df,df_11)
df = bind_rows(df,df_12)
df = bind_rows(df,df_13)

df_ = bind_rows(df_1,df_2)
df_ = bind_rows(df_,df_3)
df_ = bind_rows(df_,df_4)
df_ = bind_rows(df_,df_5)

return_words <- function(df){
  bw = df %>% count(topic,word,sort = T) %>% ungroup()
  tw = bw %>% group_by(topic) %>% summarize(total = sum(n))
  bc = left_join(bw,tw)
  print(bc)
  return(bc)}

bc <- return_words(df)

show <- function(bc){
  p <-ggplot(bc,aes(n/total,fill = topic)) + geom_histogram(show.legend = F) + xlim(NA,0.0009) + facet_wrap(~topic,ncol = 2,scales = "free_y")
  print(p)
}

show(return_words(df_))

bc.tf_idf <- bc %>% bind_tf_idf(word,topic,n);bc.tf_idf%>%print()
print(bc.tf_idf[which(bc.tf_idf$idf != 0),])
tf <- bc.tf_idf %>% select(-total) %>% arrange(desc(tf_idf))
write.csv(tf,file = "count/tf_idf_count.csv")

bc.tf_idf_ <- return_words(df_) %>% bind_tf_idf(word,topic,n)
p <- bc.tf_idf_ %>% arrange(desc(tf_idf)) %>% mutate(word = factor(word,levels= rev(unique(word)))) %>% group_by(topic) %>% top_n(5) %>% ungroup 
p1 = p %>% ggplot(aes(word,tf_idf,fill=topic)) + geom_col(show.legend = F) + labs(x = NULL,y = 'tf-idf') + facet_wrap(~topic,ncol = 2,scales = "free") + coord_flip()
print(p1)