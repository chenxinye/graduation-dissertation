#@author:chenxinye
#@2019.06.09

# LDA model bulid

library(topicmodels)
library(tidytext)
library(ggplot2)
library(dplyr)
library(tidyr)
library(magrittr)

data("AssociatedPress")
AssociatedPress
ap_lda <- LDA(AssociatedPress,k=2,control = list(seed=24))

ap_topics <- tidy(ap_lda,matrix = "beta")
ap_top_terms <- ap_topics %>% group_by(topic) %>% top_n(10,beta) %>% ungroup() %>% arrange(topic,-beta)

ap_top_terms %>% mutate(term = reorder(term,beta)) %>% ggplot(aes(term,beta,fill = factor(topic))) + geom_col(show.legend = F) +facet_wrap(~ topic,scales = "free") + coord_flip()
beta_spread <- ap_topics %>% mutate(topic = paste0("topic",topic)) %>% spread(topic,beta) %>% filter(topic1 > .001 | topic2 > .001) %>% mutate(log_ratio = log2(topic2/topic1))

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
DFC = DF %>% count(word,sort = T)
df = merge(DFC,DF,by="word")

df_SPARSE = df %>% cast_dtm(line,word,n)

ap_lda = LDA(df_SPARSE,k=25,control = list(seed=24))