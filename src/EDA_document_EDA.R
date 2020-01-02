#@author:chenxinye
#@2019.06.09

#FIRST change the document to per word per document!
#The code to count the number of each topic
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
df <- Combined_News_DJIA

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

tokens <- function(dfcol){
  #one token per document per row
  df_ = data.frame(line = 1:length(dfcol),text = dfcol)
  df_$text = as.character(df_$text)
  df_$text <- gsub("b'", "", df_$text)
  df_$text <- gsub("[0-9]", "", df_$text)
  df_ = df_%>% unnest_tokens(word, text)
  return(df_)
}

#try to eliminate stop words
dt1 = Combined_News_DJIA$Top1 %>% tokens() %>% anti_join(stop_words)
write.csv(dt1,file = "count/dt1 for showing participle.csv")
dt2 = Combined_News_DJIA$Top2 %>% tokens() %>% anti_join(stop_words)

#head(df_Top1,5)
#head(dt1,5)
return_count <- function(dt1,N = 45){
  print(dt1 %>% count(word,sort = T))
  pic = dt1 %>% count(word,sort = T) %>% filter(n >= N) %>% mutate(word = reorder(word,n)) %>% ggplot(aes(word,n)) + geom_col() + xlab(NULL) + coord_flip()
  print(pic)
  return(dt1 %>% count(word,sort = T))
}

  
csv1 = return_count(dt1,45)
csv2 = return_count(dt2,45)
write.csv(csv1,file = 'count/topic1.csv')
write.csv(csv2,file = 'count/topic2.csv')

return_freq <- function(dt1,dt2){
  m_1 = mutate(dt1,topic = 'topic 1')
  m_2 = mutate(dt2,topic = 'topic 2')
  freq <- bind_rows(m_1,m_2)
  freq <- freq %>% mutate(word = str_extract(word,"[a-z']+"))
  freq <- freq %>% count(topic,word)
  freq <- freq %>% group_by(topic)
  freq <- freq %>% mutate(proportion = n/ sum(n)) %>% select(-n) %>% spread(topic,proportion) %>% gather(topic,proportion,`topic 2`)
  freq$word = gsub("b'","",freq$word)
  return(freq)
}

frequen1_2 <- return_freq(df$Top1 %>% tokens() %>% anti_join(stop_words),df$Top2 %>% tokens() %>% anti_join(stop_words))
frequen1_3 <- return_freq(df$Top1 %>% tokens() %>% anti_join(stop_words),df$Top3 %>% tokens() %>% anti_join(stop_words))

return_show <- function(freq,str1,str2){
  p = ggplot(freq,aes(x = proportion,y=`topic 1`,color = abs(`topic 1`- proportion)))
  p = p + geom_abline(color = 'gray35',lty = 2) + geom_jitter(alpha = 0.1,size = 3.5,width = 0.3,height = 0.3)
  p = p + geom_text(aes(label = word),check_overlap = T,vjust = 1.5) + scale_x_log10(labels=percent_format()) + scale_y_log10(labels=percent_format())
  p = p + scale_color_gradient(limits = c(0,0.001),low = "darkslategray",high = 'gray85') + theme(legend.position = "none") + labs(y = str1,x = str2)
  print(p)
}
return_show(frequen1_2,'topic 1','topic 2')
return_show(frequen1_3,'topic 1','topic 3')

cor.test(data = frequen1_2[frequen1_2$topic == 'topic 2',],~proportion + `topic 1`)
cor.test(data = frequen1_3[frequen1_3$topic == 'topic 2',],~proportion + `topic 1`)

#df_all <- combine_topic(Combined_News_DJIA,26)
#dt = df_all$combine_topic %>% tokens() %>% anti_join(stop_words) %>% count(word,sort = T)
#write.csv(dt,file = "count/all_topic_count.csv")
library(wordcloud)
wordcloud(csv1$word,csv1$n,scale=c(5,1),min.freq=Inf,max.words=120,colors='#03A89E')
for (name in names(Combined_News_DJIA)[3:27]){
  de <- Combined_News_DJIA[,name] %>% tokens() %>% anti_join(stop_words) %>% return_count()
  write.csv(de,file = paste(paste('count/groupcount/topic',name),'.csv'))
}