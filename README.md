# Assignment-Data-Mining



####### CS909 -  BENIAMINO HADJ-AMAR - WEEK 10 ########




library(clusterCrit)
library(class)
library(e1071)
library(cluster)
library(class)
library(tm)
library(topicmodels)
library(SnowballC)
library(plyr)



setwd("F:/CS909/Week10")

reuters <- read.delim("reuters.tsv",
                      quote="", stringsAsFactors = FALSE)




####################################### QUESTION 1 ###########################################

#### DATA PRE - PROCESSING ####



# Function for Cleaning the Corpus 

cleanCorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removeNumbers)
  corpus.tmp <- tm_map(corpus.tmp, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, content_transformer(tolower)) 
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  corpus.tmp <- tm_map(corpus.tmp, stemDocument, language = "english")  
  return(corpus.tmp)
}


# Vector of 21578 documents
docs <- reuters[-13804, 140]

# Creating Corpus
s.cor <- Corpus(VectorSource(docs))

# Cleaning Corpus
s.cor.cl <- cleanCorpus(s.cor)
inspect(s.cor.cl)

# Creating Term Document Matrix
s.tdm <- TermDocumentMatrix(s.cor.cl)

# Removing Sparse Term
s.tdm.s <- removeSparseTerms(s.tdm, 0.95) # 178 words
str(s.tdm.s)

# Basically, representing Term Document Matrix (rows= n.of document, columns= terms)
s.mat <- t(data.matrix(s.tdm.s))
s.df <- as.data.frame(s.mat, stringsAsFactors=FALSE)

# So we have term counts:
s.df
 



# Function TF*IDF

tfidf <- function(tab) {
  tf <- tab
  id <- function(col) { sum(!col==0) }
  idf <- log(nrow(tab)/apply(tab, 2, id))
  tfidf <- tab
  for(word in names(idf)){ tfidf[, word] <- tf[, word]*idf[word] }
  return(tfidf)
}


# Creating appropriate weighting using TF*IDF
s.df.final <- tfidf(s.df)










############################## QUESTION 2 ###################################################




### FOCUSING ON THE 10 MOST POPULAR TOPICS




# Considering only 10 most popular classes (row 13804 is NA)

reuters.mod <- reuters[-13804, c(
  which(names(reuters)=="topic.earn"),
  which(names(reuters)=="topic.acq"),
  which(names(reuters)=="topic.money.fx"),
  which(names(reuters)=="topic.grain"),
  which(names(reuters)=="topic.crude"),
  which(names(reuters)=="topic.trade"),
  which(names(reuters)=="topic.interest"),
  which(names(reuters)=="topic.ship"),
  which(names(reuters)=="topic.wheat"),
  which(names(reuters)=="topic.corn") )]


reuters.mod



# Creating for each document its relevant topic

topic <- c()

for(i in 1:nrow(reuters.mod)) {
  if(sum(reuters.mod[i,]) == 0) {
    topic[i] <- NA}
  if(sum(reuters.mod[i, ]) == 1) {
    topic[i] <- names(reuters.mod)[which(reuters.mod[i, ]==1)] }
  if(sum(reuters.mod[i, ]) > 1) {
    topic[i] <- names(reuters.mod)[sample(which(reuters.mod[i, ]==1), 1)]} # RANDOMLY SELECTED BETWEEN TWO
}

s.df.count <- cbind(s.df, topic)
s.df.mod <- cbind(s.df.final, topic)

# Identifying missing values ( we consider only 10 most popular topics)
NA.rows <- which(is.na(topic))

# Creating a data frame with documents, terms, and rispective topic
s.df.topic <- s.df.mod[-NA.rows, ]
s.df.topic.count <- s.df.count[-NA.rows, ]

# Deleting rows with zero
s.df.topic <- s.df.topic[apply(s.df.topic[, -179], 1, function(x) !all(x==0)),]
s.df.topic.count <- s.df.topic.count[apply(s.df.topic.count[, -179], 1, function(x) !all(x==0)), ]



# Therefore we obtained one feature representation of the documents,
# by using a "Bag-of-words" approach. Its representation corresponds to
# the data.frame "s.df.topic".




############## BAG.OF.WORDS #################
s.df.topic
#############################################





# Another representation it's obtained by considering topic models. By using LDA 
# (Latent Dirichlet Association) we have another feature representation for these documents.



k <- 10 # Number of topic

# Control parameter for convergence of the Variational E-M for the allocation of the documents
# in the several topics.
control_LDA_VEM <- list(estimate.alpha = TRUE, alpha = 50/k, estimate.beta = TRUE,
                        verbose = 0, prefix = tempfile(), save=0, keep=0,
                        seed = as.integer(Sys.time()), nstart=1, best=TRUE,
                        var = list(iter.max=500, tol=10^-6),
                        em = list(iter.max = 1000, tol = 10^-4),
                        initialize = "random")


# Fitting the model to obtain the new features
mod.lda <- LDA(x = s.df.topic.count[, -179], k = k, method = "VEM", control = control_LDA_VEM)


# Here's the the new data.frame with the new features, namely
# a dataframe with for each document, whe have 10 variables. 
# Each variable represents the probability that this document belongs to 
# its relative topic. It's a sort of Bayesian Mixture over the topics. 


lda.features <- cbind(as.data.frame(mod.lda@gamma), s.df.topic.count[, 179])
colnames(lda.features)[11] <- "topic"



########### TOPIC MODELS - LATENT DIRICHLET ALLOCATION #####################
lda.features
##########################################################################






# Finally we propose a combined approach using both bag of words & topic probabilities.

bow.lda.features <- cbind(s.df.topic[, -179], lda.features)
colnames(bow.lda.features)[189] <- "topic"




########## LDA + BAG OF WORDS ####################
bow.lda.features
##############################################










################################# QUESTION 3 ################################################





##### 10-fold Cross-Validation for Naive Bayes, SVM, and kNN




# Firstly we shuffle our data
s.topic <- s.df.topic[sample(nrow(s.df.topic)), ]
s.topic.lda <- lda.features[sample(nrow(lda.features)), ]
s.topic.bow.lda <- bow.lda.features[sample(nrow(bow.lda.features)), ]

# Create 10 equally size folds
folds <- cut(seq(1,nrow(s.topic)), breaks=10, labels=FALSE)


# Initializing some useful things

# Bag of Words
conf10F.SVM <- 0
acc10F.SVM <- c()

conf10F.nB <- 0
acc10F.nB <- c()

conf10F.kNN <- 0
acc10F.kNN <- c()

# LDA
lda.conf10F.SVM <- 0
lda.acc10F.SVM <- c()

lda.conf10F.nB <- 0
lda.acc10F.nB <- c()

lda.conf10F.kNN <- 0
lda.acc10F.kNN <- c()

# Bag of Words + LDA
bow.lda.conf10F.SVM <- 0
bow.lda.acc10F.SVM <- c()

bow.lda.conf10F.nB <- 0
bow.lda.acc10F.nB <- c()

bow.lda.conf10F.kNN <- 0
bow.lda.acc10F.kNN <- c()





## Performing Cross Validation for each Classifier



########## BAG OF WORDS


for(i in 1:10) {
  
  
  # Splitting dataset in training and test set
  testIndexes <- which(folds==i, arr.ind=TRUE)
  testData <- s.topic[testIndexes, ]
  trainData <- s.topic[-testIndexes, ]
  trainTopic <- s.topic$topic[-testIndexes]
  
  # Performing classification naive Bayes
  reuters.nB <- naiveBayes(topic ~ ., data=trainData)
  pred.nB <- predict(reuters.nB, testData[, -179])
  tab.nB <- table(testData$topic, pred.nB)
  conf10F.nB <- conf10F.nB + tab.nB
  acc10F.nB[i] <- sum(diag(tab.nB))/nrow(testData)
  
  # Performing classification SVM
  reuters.SVM <- svm(topic ~ ., data = trainData)
  pred.SVM <- predict(reuters.SVM, testData[, -179])
  tab.SVM <- table(testData$topic, pred.SVM)
  conf10F.SVM <- conf10F.SVM + tab.SVM
  acc10F.SVM[i] <- sum(diag(tab.SVM))/nrow(testData)
  
  # Performing Classification kNN
  pred.reuters.kNN <- knn(trainData[, -179], testData[, -179], trainTopic, 10)
  tab.kNN <- table(testData$topic, pred.reuters.kNN )
  conf10F.kNN <- conf10F.kNN + tab.kNN
  acc10F.kNN[i] <- sum(diag(tab.kNN))/nrow(testData)
  
    
}



########## LDA


for(i in 1:10) {
  
  
  # Splitting dataset in training and test set
  testIndexes <- which(folds==i, arr.ind=TRUE)
  testData <- lda.features[testIndexes, ]
  trainData <- lda.features[-testIndexes, ]
  trainTopic <- lda.features$topic[-testIndexes]
  
  # Performing classification naive Bayes
  reuters.nB <- naiveBayes(topic ~ ., data=trainData)
  pred.nB <- predict(reuters.nB, testData[, -11])
  tab.nB <- table(testData$topic, pred.nB)
  lda.conf10F.nB <- lda.conf10F.nB + tab.nB
  lda.acc10F.nB[i] <- sum(diag(tab.nB))/nrow(testData)
  
  # Performing classification SVM
  reuters.SVM <- svm(topic ~ ., data = trainData)
  pred.SVM <- predict(reuters.SVM, testData[, -11])
  tab.SVM <- table(testData$topic, pred.SVM)
  lda.conf10F.SVM <- lda.conf10F.SVM + tab.SVM
  lda.acc10F.SVM[i] <- sum(diag(tab.SVM))/nrow(testData)
  
  # Performing Classification kNN
  pred.reuters.kNN <- knn(trainData[, -11], testData[, -11], trainTopic, 10)
  tab.kNN <- table(testData$topic, pred.reuters.kNN )
  lda.conf10F.kNN <- lda.conf10F.kNN + tab.kNN
  lda.acc10F.kNN[i] <- sum(diag(tab.kNN))/nrow(testData)
  
}






########## BAG OF WORDS + LDA


for(i in 1:10) {
  
  
  # Splitting dataset in training and test set
  testIndexes <- which(folds==i, arr.ind=TRUE)
  testData <- bow.lda.features[testIndexes, ]
  trainData <- bow.lda.features[-testIndexes, ]
  trainTopic <- bow.lda.features$topic[-testIndexes]
  
  # Performing classification naive Bayes
  reuters.nB <- naiveBayes(topic ~ ., data=trainData)
  pred.nB <- predict(reuters.nB, testData[, -189])
  tab.nB <- table(testData$topic, pred.nB)
  bow.lda.conf10F.nB <- bow.lda.conf10F.nB + tab.nB
  bow.lda.acc10F.nB[i] <- sum(diag(tab.nB))/nrow(testData)
  
  # Performing classification SVM
  reuters.SVM <- svm(topic ~ ., data = trainData)
  pred.SVM <- predict(reuters.SVM, testData[, -189])
  tab.SVM <- table(testData$topic, pred.SVM)
  bow.lda.conf10F.SVM <- bow.lda.conf10F.SVM + tab.SVM
  bow.lda.acc10F.SVM[i] <- sum(diag(tab.SVM))/nrow(testData)
  
  # Performing Classification kNN
  pred.reuters.kNN <- knn(trainData[, -189], testData[, -189], trainTopic, 10)
  tab.kNN <- table(testData$topic, pred.reuters.kNN )
  bow.lda.conf10F.kNN <- bow.lda.conf10F.kNN + tab.kNN
  bow.lda.acc10F.kNN[i] <- sum(diag(tab.kNN))/nrow(testData)
  
}












## BAG OF WORDS ## 

# Confusion Matrix summed over 10 fold cross validation 
conf10F.nB # Naive Bayes
conf10F.SVM # SVM
conf10F.kNN # kNN


# Accuracy of each fold 
acc10F.nB # Naive Bayes
acc10F.SVM # SVM
acc10F.kNN # kNN


# Means and Standard Deviations
mean(acc10F.nB); sd(acc10F.nB) # naive Bayes
mean(acc10F.SVM); sd(acc10F.SVM) # SVM
mean(acc10F.kNN); sd(acc10F.kNN) # kNN




## LDA ##

# Confusion Matrix summed over 10 fold cross validation 
lda.conf10F.nB # Naive Bayes
lda.conf10F.SVM # SVM
lda.conf10F.kNN # kNN

# Accuracy of each fold 
lda.acc10F.nB # Naive Bayes
lda.acc10F.SVM # SVM
lda.acc10F.kNN # kNN

# Means and Standard Deviations
mean(lda.acc10F.nB); sd(lda.acc10F.nB) # naive Bayes
mean(lda.acc10F.SVM); sd(lda.acc10F.SVM) # SVM
mean(lda.acc10F.kNN); sd(lda.acc10F.kNN) # kNN




## BAG OF WORDS + LDA ##

# Confusion Matrix summed over 10 fold cross validation 
bow.lda.conf10F.nB # Naive Bayes
bow.lda.conf10F.SVM # SVM
bow.lda.conf10F.kNN # kNN

# Accuracy of each fold 
bow.lda.acc10F.nB # Naive Bayes
bow.lda.acc10F.SVM # SVM
bow.lda.acc10F.kNN # kNN

# Means and Standard Deviations
mean(bow.lda.acc10F.nB); sd(bow.lda.acc10F.nB) # naive Bayes
mean(bow.lda.acc10F.SVM); sd(bow.lda.acc10F.SVM) # SVM
mean(bow.lda.acc10F.kNN); sd(bow.lda.acc10F.kNN) # kNN





#### Auxiliary Function for Evaluating Performance Measures

performance <- function(tab) {
  
  
  Recall <- c()
  Precision <- c()
  
  
  #Recall
  for(i in 1: nrow(tab)) {
    Recall[i] <- tab[i,i]/(sum(tab[i, ]))
  }
  
  #Precision
  for(i in 1: nrow(tab)) {
    Precision[i] <- tab[i,i]/(sum(tab[, i]))
  }
  
  for( i in 1:nrow(tab)) {
    
    if(is.na(Recall[i])) {Recall[i] <- 0}
    else if (is.na(Precision[i])) {Precision[i] <- 0}
    
  }
  
  
  
  # F-Measure
  F <- (2*Precision*Recall)/(Precision + Recall)
  
  
  # Accuracy
  Accuracy <- sum(diag(tab))/nrow(s.df.topic)
  
  # Macro Average Recall
  M.Recall <- sum(Recall)/nrow(tab)
  
  # Macro Average Precision
  M.Precision <- sum(Precision)/nrow(tab)
  
  # micro Average Recall
  num <- sum(diag(tab))
  den.r <- 0
  for(i in 1:nrow(tab)) {
    temp.r <- sum(tab[i,])
    den.r <- den.r + temp.r
  }
  m.Recall <- num/den.r
  
  
  # micro Average Precision
  den.p <- 0
  for(i in 1:nrow(tab)){
    temp.p <- sum(tab[, i])
    den.p <- den.p + temp.p
  }
  m.Precision <- num/den.p
  
  
  # Rounding & Processing
  Recall <- round(Recall, digits=5) 
  Precision <- round(Precision, digits=5)
  F <- round(F, digits=5)
  type <- row.names(tab)
  type <- as.data.frame(type)
  
  
  return(list(ClassMeasures = cbind(type, Recall, Precision, F),
              Accuracy = Accuracy,
              M.Recall = M.Recall,
              M.Precision = M.Precision,
              m.Recall = m.Recall,
              m.Precision =  m.Precision))
  
}



# Condifence Interval for Accuracy of SVM

a <- performance(conf10F.SVM)$Accuracy; a

conf.int <- c( a - 1.96* sqrt( (a*(1-a)) / nrow(s.df.topic) ),
               a + 1.96* sqrt( (a*(1-a)) / nrow(s.df.topic) ) ); conf.int




### PERFORMANCE MEASURES ###




#### BAG-OF WORDS

# Performance measures for SVM
conf10F.SVM
performance(conf10F.SVM)

# Performance measures for naiveBayes
conf10F.nB
performance(conf10F.nB)

# Performance measures for kNN
conf10F.kNN
performance(conf10F.kNN)




#### LDA

# Performance measures for SVM
lda.conf10F.SVM
performance(lda.conf10F.SVM)

# Performance measures for naiveBayes
lda.conf10F.nB
performance(lda.conf10F.nB)

# Performance measures for kNN
lda.conf10F.kNN
performance(lda.conf10F.kNN)




#### BAG OF WORDS + LDA

# Performance measures for SVM
bow.lda.conf10F.SVM
performance(bow.lda.conf10F.SVM)

# Performance measures for naiveBayes
bow.lda.conf10F.nB
performance(bow.lda.conf10F.nB)

# Performance measures for kNN
bow.lda.conf10F.kNN
performance(bow.lda.conf10F.kNN)











####################################### QUESTION 4 ###########################################



# Firstly I need to understand which topics are really not empty, in order to
# understand how many effective possibile topics these documents belong to. 

reuters.work <- reuters[-13804, - c(1, 2, 3, 139, 140)]
emptycol <- c()
for( i in 1:ncol(reuters.work)) {
  if(sum(reuters.work[, i]) == 0 ) {emptycol <- c(emptycol, i)}
}

reuters.work <- reuters.work[, -emptycol]

# So I discovered that there are 118 possible topics

ncol(reuters.work)





## Preparing Data for Clustering

 
s.df.clust <- s.df[apply(s.df, 1, function(x) !all(x==0)),  ]


# Normalising function (euclidean)
norm_eucl <- function(m) {
  m/apply(m, 1, function(x) sum(x^2)^.5)
}

df.clust.norm <- norm_eucl(s.df.clust)



# Evaluating dissimilary matrix
dissimilarity <- dist(df.clust.norm)



####### HIERARCHICAL CLUSTERING

mod.hclust <- hclust(dissimilarity, method="ward.D2")


#Plotting dendogram
plot(mod.hclust, xlab=NA, sub=NA)
rect.hclust(mod.hclust, k = 10, border = "red")


######## K MEANS
mod.kmeans <- kmeans(df.clust.norm, centers = 118)

#Plotting clusters on the principal components
plot(prcomp(df.clust.norm)$x, col=mod.kmeans$cl, main="kMeans")


#### PAM

mod.pam <- pam(dissimilarity, k=10, diss=T)

#Plotting clusters on the princiapl components
plot(prcomp(df.clust.norm)$x, col=mod.pam$cluster, main= "PAM")



### EM
mix.model <- Mclust(df.clust.norm)


# Evaluation clusters


### SILHOUETTE
sil.hclust <- silhouette(cutree(mod.hclust, 11), dissimilarity)
plot(sil.hclust, main="Silhouette Hclust")


# Silhouette and Dunn Index

intIdx.kmeans <- intCriteria(as.matrix(df.clust.norm), mod.kmeans$cluster, c("Silhouette", "Dunn"))
intIdx.pam <- intCriteria(as.matrix(df.clust.norm), mod.pam$cluster, c("Silhouette", "Dunn")) 
