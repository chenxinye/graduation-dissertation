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

WR = DF %>% group_by(word) %>% filter(n() >= 20) %>% pairwise_cor(word,line,sort = T)
d = WR[which(WR$correlation <1),]
write.csv(d,file = "corr.csv")

top_word <- c("government","people","police","world","war","israel","u.s","killed","president")

d %>% filter(item1 %in% top_word) %>% group_by(item1) %>% top_n(6) %>% mutate(item2 = reorder(item2,correlation)) %>% 
  ggplot(aes(item2,correlation)) + geom_bar(stat = "identity") + facet_wrap(~ item1,scales = "free") + coord_flip()

#d %>% filter(correlation > 0.8) %>% graph_from_data_frame() %>% ggraph(layout = "auto") + 
 # geom_edge_link(aes(edge_alpha =correlation),show.legend = F) + geom_node_point(color = "lightblue",size = 5) + 
  #geom_node_text(aes(label = name),repel = T) + theme_void()

return_summary <- function(comment_word,sup = 0.1,con = 0.1,boole = T){
  table_aprori <- as(split(comment_word$word,comment_word$line),"transactions")
  model_build <- apriori(table_aprori,parameter = list(support = sup,confidence=con))
  output <- inspect(sort(model_build,by = "confidence"));summary(model_build)
  output <- output[-which(output$lhs == "{}"),];head(output,30)
  if (boole == T){return (output)
  }else{return (model_build)}}

sc = return_summary(DF) # calculate support and confidence

return_writer <- function(df,string){
  write.csv(df, string, row.names = FALSE)}

return_writer(sc,"count/support and confidence.csv")

sc_ = return_summary(DF,sup = 0.1,con = 0.1,boole = F) # calculate support and confidence
output_neg <- head(sort(sc_,by = "confidence"),30)
plot(output_neg,method = "grouped",measure = "lift",shading ="support")
