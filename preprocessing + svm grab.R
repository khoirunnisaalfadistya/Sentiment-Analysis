setwd('D:/KULIAH/SEMESTER 7/SKRIPSI/DATA/grab')

#LOAD PACKAGE#
library(tm)
library(stringr)
library(dplyr)

#Ambil data#
dok1=read.csv('data grab full.csv',sep=';',header = TRUE)
glimpse(dok1)
dok=dok1$dokumen


#Merubah csv ke Vector Corpus#
corpusdok=Corpus(VectorSource(dok))
inspect(corpusdok)

###LANGKAH PREPROCESSING###
#Cleaning hapus URL#
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
dok_URL <- tm_map(corpusdok, content_transformer(removeURL))
inspect(dok_URL)

remove.mention <- function(x) gsub("@\\S+", "", x)
dok_mention <- tm_map(dok_URL, remove.mention)
inspect(dok_mention)

remove.hashtag <- function(x) gsub("#\\S+", "", x)
dok_hashtag <- tm_map(dok_mention, remove.hashtag)
inspect(dok_hashtag)

remove.emoticon <- function (x) gsub("[^\x01-\x7F]", "", x)
dok_emoticon <- tm_map(dok_hashtag,remove.emoticon)
inspect(dok_emoticon)

remove.code <- function (x) gsub("<\\S+","",x)
dok_code <- tm_map(dok_emoticon,remove.code)
inspect(dok_code)

#remove punctuation#
dok_punctuation <- tm_map(dok_code, content_transformer(removePunctuation))
inspect(dok_punctuation)

#remove whitespace#
dok_whitespace<-tm_map(dok_punctuation,stripWhitespace)
inspect(dok_whitespace)

#remove number#
dok_nonumber <- tm_map(dok_whitespace, content_transformer(removeNumbers))
inspect(dok_nonumber)
hasil.cleansing<-data.frame(dok,text=unlist(sapply(dok_nonumber, '[')), stringsAsFactors=F)
write.csv(hasil.cleansing,'hasil-cleansing.csv')


#case folding#
dok_casefolding <- tm_map(dok_nonumber, content_transformer(tolower))
inspect(dok_casefolding)

#remove duplicate character
remove.char <- function (x) gsub("([[:alpha:]])\\1{2,}", "\\1",x)
dok_char <- tm_map(dok_casefolding,remove.char)
inspect(dok_char)
hasil.casefold<-data.frame(dok,text=unlist(sapply(dok_char, '[')), stringsAsFactors=F)
write.csv(hasil.casefold,'hasil-casefold.csv')

#Slang Word
#load slangword#
slang <- read.csv("slangword_list grab.csv", header=T)
old_slang <- as.character(slang$old) 
new_slang <- as.character(slang$new)
slangword <- function(x) Reduce(function(x,r) gsub(slang$old[r],slang$new[r],x,fixed=F),
                                seq_len(nrow(slang)),x)
dok_slangword <- tm_map(dok_char,slangword)
inspect(dok_slangword)
hasil.slangwords<-data.frame(dok,text=unlist(sapply(dok_slangword, '[')), stringsAsFactors=F)
write.csv(hasil.slangwords,'hasil-slangwords.csv')


#remove stopwords#
swindo<-as.character(readLines("stopwords grab.csv"))
dok_stopword<-tm_map(dok_slangword,removeWords,swindo)
inspect(dok_stopword)
hasil.stopwords<-data.frame(dok,text=unlist(sapply(dok_stopword, '[')), stringsAsFactors=F)
write.csv(hasil.stopwords,'hasil-stopwords.csv')

#STEMMING#
library(NLP)
library(tau)
library(katadasaR)
library(parallel)
stem_text<-function(text,mc.cores=1)
{
  stem_string<-function(str)
  {
    str<-tokenize(x=str)
    str<-sapply(str,katadasaR)
    str<-paste(str,collapse = "")
    return(str)
  }
  x<-mclapply(X=text,FUN=stem_string,mc.cores=mc.cores)
  return(unlist(x))
}
dok_stemming<-tm_map(dok_stopword,stem_text)
dok_stemming<-tm_map(dok_stemming,stripWhitespace)
inspect(dok_stemming)
hasilstemming<-data.frame(dok1,text=unlist(sapply(dok_stemming, '[')), stringsAsFactors=F)
write.csv(hasilstemming, file="hasil stemming terakhir.csv")


#TDM#
tdm<-TermDocumentMatrix(dok_stemming)
tdm2<-weightTfIdf(tdm,normalize = TRUE)
inspect(tdm)
inspect(tdm2)
m<-as.matrix(tdm)
m2<-as.matrix(tdm2)

write.csv(m, file="m.csv")
write.csv(m2, file="m2.csv")
v<-sort(rowSums(m),decreasing = TRUE)
d<-data.frame(word=names(v),freq=v)
write.csv(d,'frekuensi kata.csv')
head(d,20)

#Word Cloud#
library(wordcloud2)
wordcloud2(d,size = 1,fontFamily = 'Segoe UI',color = "random-dark")

#SVM
library(e1071)
library(caret)

dtm2<-DocumentTermMatrix(dok_stemming)
dtm3<-weightTfIdf(dtm2,normalize = TRUE)
dtm<-as.matrix(dtm3)
sentimen<-dok1$sentimen
datasvm<-data.frame(sentimen,dtm,stringsAsFactors = F)


training=datasvm[1:376,]
testing = datasvm[377:470, ]


svmfit = svm(sentimen ~ ., data = training, kernel = "linear", scale = FALSE)
confusionMatrix(svmfit$fitted,training[,1])

prediksiSVM=predict(svmfit,testing[,-1],type='class')
confusionMatrix(prediksiSVM,testing$sentimen)


#HASIL
support.vector<-as.matrix(svmfit$SV)
write.csv(support.vector,'support vector.csv')
coefs<-as.matrix(abs(svmfit$coefs))
write.csv(coefs,'koefisien lagrange.csv')
fit<-as.matrix(svmfit$fitted)
write.csv(fit,'fit.csv')
decission.value<-as.matrix(svmfit$decision.values)
write.csv(decission.value,'decission value.csv')
pred<-as.matrix(prediksiSVM)
write.csv(pred,'prediksi testing.csv')

svv<-read.csv('support vector.csv',sep=";",header = T)
svv1<-svv$dok
msv<-as.matrix(dok1)
msv1<-as.matrix(msv[svv1,])
msv2<-cbind(svv1,msv1)
write.csv(msv2,'matrix support vector.csv')
