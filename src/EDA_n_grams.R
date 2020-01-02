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
set.seed(200)

data(stop_words)
setwd("D:/graduation project/src/data")
Combined_News_DJIA <- read.csv("Combined_News_DJIA.csv")
#the following steps are to do the n-grams method
#delete stopwords
return_ngrams <- function(Combined_News_DJIA){
  d = combine_topic(Combined_News_DJIA,26)
  bigrams = data.frame(line = 1:length(d$combine_topic),topic = as.vector(d$combine_topic))
  bigrams = bigrams %>% unnest_tokens(bigram,topic,token= 'ngrams',n=2)
  bigrams$bigram = gsub("b'","",bigrams$bigram)
  return(bigrams)
}

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

bigrams <- return_ngrams(Combined_News_DJIA)
bigrams_sep <- bigrams %>% separate(bigram,c("word1","word2"),sep = " ")

#eliminate stopwords
bif <- bigrams_sep %>% filter(!word1 %in% stop_words$word) %>% filter(!word2 %in% stop_words$word)
bif_counts <- bif %>% count(word1,word2,sort = T)
print(bif_counts)
write.csv(bif_counts,file = "count/grams_count.csv")
AFINN <- get_sentiments("afinn")

#after confirming that the text do not contain "not", I skip the step to do another analysis
nw <- bif_counts %>% filter(word1 == "not") %>% count(word1,word2,sort = T)
bg <- bif_counts %>% filter(n > 35) %>% graph_from_data_frame()
a <- grid::arrow(type = "closed",length = unit(.15,"inches"))
ggraph(bg,layout = "fr") + geom_edge_link() + geom_node_point() + geom_node_text(aes(label = name),vjust = 1,hjust = 1)