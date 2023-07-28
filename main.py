# import necessary modules
import requests, json, os, datetime, logging

import nltk

from pypdf import PdfReader

from io import BytesIO

from azure.storage.blob import BlobServiceClient, BlobClient

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

import modules.functions as fn
import modules.config as cfg

# define key variables
searchEndpoint = cfg.searchEndpoint
searchKey = cfg.searchKey
searchIndex = cfg.searchIndex

openAiEndpoint = cfg.openAiEndpoint
openAiKey = cfg.openAiKey
baseAIModel = cfg.openAiDeployedModel
embedAIModel = cfg.openAIEmbedModel

deployedModels = fn.fnGetDeployedModels( openAiEndpoint, openAiKey )

# print( deployedModels )

for model in deployedModels :

    print( "Deployed Model : ", model[ "id" ], " based on model : ", model[ "model" ] )

masterPromptList = []

# define connections
# storage account connection
jsonResultList = fn.fnSeachIndex( searchEndpoint, searchKey, searchIndex )

# print( jsonResultList[ "value" ] )

contentList = jsonResultList[ "value" ]

for item in contentList :

    currFileText = item[ "merged_content" ]

    # generate document embeddings
    embeddingJson = fn.fnGenerateEmbeddings( currFileText, embedAIModel, openAiEndpoint, openAiKey )

    # print( embeddingJson )

    pageSentenceList = fn.fnSplitPage( currFileText )

    for sentence in pageSentenceList :

        masterPromptList.append( sentence )

print( masterPromptList )

# Function for generating prompts from list of embeddings - needs some work
# masterOutputList = fn.fnPromptOpenAI( masterPromptList, openAiEndpoint, openAiKey, baseAIModel )

# leaving model tuning disabled until available

# create tuning file
# tuningFileId = fn.fnCreateTuningFile( masterPromptList, openAiEndpoint, openAiKey, storageEndpoint, storageSasToken )

# create tuning job
# jsonJobResponse = fn.fnModelFineTune( tuningFileId, openAiEndpoint, openAiKey, baseAIModel )

# print( str( jsonJobResponse ) )