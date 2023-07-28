import os, json, requests, logging, datetime, uuid

from IPython.display import Image

import nltk

import modules.config as cfg

from azure.storage.blob import BlobServiceClient, BlobClient

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

OneOrTwo = lambda x : "0" + str( x ) if x < 10 else str( x )

nltk.download( "punkt" )


# fnSearchIndex : pull document contents from Cognitive Search index for generating embeddings
# using Cognitive Search to perform OCR on documents
def fnSeachIndex( searchEndpoint, searchKey, searchIndex ) :

    requestUrl = searchEndpoint + "/indexes/" + searchIndex + "/docs/search?api-version=2020-06-30"

    headers = {
        "content-type": "application/json",
        "api-key": searchKey
    }

    params = {
        "queryType" : "simple",
        "select" : "metadata_storage_path, merged_content",
    }

    apiRequest = requests.post( requestUrl, headers = headers, json = params )

    apiResponse = apiRequest.json()

    # print( apiResponse )

    return apiResponse


# fnSplitPage : uses NLTK to vectorize page by sentence
def fnSplitPage( pageText ) :

    # nltk.download( "punkt" )

    # punctuationList = [ ".", "?", "!" ]
    removeList = [ "\n", "\r", "\t" ]

    for item in removeList :

        pageText = pageText.replace( item, "" )
    
    # split page into sentences
    pageSentences = nltk.tokenize.sent_tokenize( pageText )

    # remove empty sentences
    pageSentences = [ sentence for sentence in pageSentences if sentence != "" ]

    # return list of sentences
    return pageSentences


# fnGetDeployedModels : provide ability to pull list of deployed models
def fnGetDeployedModels( endpoint, key ) :

    requestUrl = endpoint + "openai/deployments?api-version=2022-12-01"

    headers = {
        "content-type": "application/json",
        "api-key": key
    }

    apiRequest = requests.get( requestUrl, headers = headers )

    apiResponse = json.loads( apiRequest.text )

    # jsonResponse = json.dumps( apiResponse[ "data" ], indent = 4 )

    modelList = apiResponse[ "data" ]

    return modelList


# fnCreateTuningFile : import list of prompts and write to a JSONL file in Blob Storage
def fnCreateTuningFile( hints, aiEndpoint, aiKey, blobEndpoint, blobKey ) :

    # import created hintfile into Azure Blob Storage

    containerList = []

    containerName = "hint-files"

    currDateTime = datetime.datetime.now()

    currYear = str( currDateTime.year )
    currMonth = OneOrTwo( currDateTime.month )
    currDay = OneOrTwo( currDateTime.day )
    currHour = OneOrTwo( currDateTime.hour )
    currMinute = OneOrTwo( currDateTime.minute )
    currSecond = OneOrTwo( currDateTime.second )

    hintFile = "prompts_" + currYear + currMonth + currDay + "-" + currHour + currMinute + currSecond + ".jsonl"

    requestUrl = aiEndpoint + "openai/files/import?api-version=2022-12-01"

    headers = {
        "api-key": aiKey
    }

    # create and upload hint(s) file to blob storage
    blobSvcConn = BlobServiceClient( account_url = blobEndpoint, credential = blobKey )
    contConn = blobSvcConn.get_container_client( container = containerName )

    existingContainers = blobSvcConn.list_containers()

    for container in existingContainers :

        containerList.append( container[ "name" ] )

    if containerName in containerList :

        contConn = blobSvcConn.get_container_client( container = containerName )

    else :

        contConn = blobSvcConn.create_container( name = containerName )

    with open( hintFile, "at" ) as file :

        for hint in hints :

            hintBlock = { "prompt" : "Act as an investigator. You are attempting to solve the murder of John F. Kennedy. Add this phrase to the body of evidence to determine who committed the murder of John F. Kennedy.", "completion" : hint }

            file.write( hintBlock + "\n" )

    # upload file to blob storage
    blobConn = blobSvcConn.get_blob_client( container = containerName, blob = hintFile )
    blobConn.upload_blob( hintFile, overwrite = True )

    fileUrl = blobEndpoint + containerName + "/" + hintFile + blobKey

    # return fileUrl

    # import file, get file id

    jsonBody = {
        "content_url" : fileUrl ,
        "filename" : hintFile ,
        "purpose" : "fine-tune"
    }

    response = requests.post( requestUrl, headers = headers, json = jsonBody )

    parseResponse = response.json()

    print( "parseResponse: " + str( parseResponse ) )

    fileId = parseResponse[ "id" ]

    return fileId


# fnModelFineTune : create job to import file from Blob Storage and create model fine-tuning job
def fnModelFineTune( fileId, endpoint, key, model ) :

    jobResponse = {}

    requestUrl = endpoint + "openai/fine-tunes?api-version=2022-12-01"

    headers = {
        "api-key": key
    }

    jsonBody = {
        "model": model,
        "training_file" : fileId
    }

    response = requests.post( requestUrl, headers = headers, json = jsonBody )

    fineTuneResponse = response.json()

    print( "fineTuneResponse: " + str( fineTuneResponse ) )

    jobId = fineTuneResponse[ "training_files" ][ 0 ][ "id" ]
    jobStatus = fineTuneResponse[ "training_files" ][ 0 ][ "status"]

    jobResponse[ "jobId" ] = jobId
    jobResponse[ "jobStatus" ] = jobStatus

    return( json.loads( jobResponse ) )

# fnPromptOpenAI : send collection of prompts, and get results for testing
# needs to be reworked - doesn't really accomplish anything

def fnPromptOpenAI( promptList, endpoint, key, model ) :

    responseList = {}
    responseList[ "responses" ] = []

    requestUrl = endpoint + "openai/deployments/" + model + "/completions?api-version=2022-12-01"

    headers = {
        "content-type": "application/json",
        "api-key": key
    }

    for prompt in promptList :
    
        payload = {
            "prompt" : prompt
        }

        apiRequest = requests.post( requestUrl, headers = headers, json = payload )

        apiResponse = json.loads( apiRequest.text )

        jsonResponse = json.dumps( apiResponse, indent = 4 )

        # print( jsonResponse )


# function to display image file
def fnDisplayImage( imageFile ) :

    return Image( filename = imageFile )


# generate embeddings for a given page
def fnGenerateEmbeddings( text, model, endpoint, key ) :

    # create user guid
    userID = uuid.uuid4()

    # create embedding URL
    requestUrl = endpoint + "openai/deployments/" + model + "/embeddings?api-version=2022-12-01"

    headers = {
        "content-type": "application/json" ,
        "api-key": key
    }

    body = {
        "input" : text ,
        "user" : str( userID )
    }

    apiRequest = requests.post( requestUrl, headers = headers, json = body )

    apiResponse = apiRequest.json()

    jsonResponse = json.dumps( apiResponse, indent = 4 )

    return jsonResponse