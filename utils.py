#导入模型
from langchain_openai import  ChatOpenAI
#导入PDF加载器
from langchain_community.document_loaders import  PyPDFLoader
#导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
#导入嵌入模型
from langchain_openai import OpenAIEmbeddings
#导入向量数据库
from langchain_community.vectorstores import FAISS
#导入对话检索脸
from langchain.chains import  ConversationalRetrievalChain

#定义一个函数 参数包括大模型的api_key 记忆和上传的文件、问题
def qa_agent(openai_api_key, memory, uploaded_file, question):
    #定义模型的模型
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="sk-z65dHrAmRvNSxh9NC690770e1060419bB396C1A94fCfD629", base_url= "https://api.aigc369.com/v1", max_tokens = 1000)


    #1、加载上传的文件
    # 对上传的内容进行读取 读出来的是二进制内容
    file_content = uploaded_file.read()
    #临时文件路径
    temp_file_path = "temp.pdf"
    #将读取出来的二进制内容进行写入
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    #创建加载器实列
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    #2、对文档进行分割
    #实例化一个分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )

    #调用文件分割方法,传入前面的文档列表
    texts = text_splitter.split_documents(docs)
    #3、做向量嵌入
    #创建一个嵌入模型的实列化对象
    embeddings_modle = OpenAIEmbeddings(openai_api_key="sk-z65dHrAmRvNSxh9NC690770e1060419bB396C1A94fCfD629", base_url= "https://api.aigc369.com/v1",max_tokens=1000)

    #把文档内容向量化

    #4、把向量化后的文本存储到向量数据库
    db = FAISS.from_documents(texts, embeddings_modle)
    #得到一个检索器
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
#使用链实现检索增强生成
    response = qa.invoke({"chat_history": memory, "question": question})
    return response