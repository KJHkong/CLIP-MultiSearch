import torch
import clip
from PIL import Image

def main():
    device = "cpu"  
    #加载CLIP模型和预处理函数
    # "ViT-B/32"表示使用Vision Transformer Base架构，patch大小为32x32
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 1) 读入图片并预处理
    '''
    # Image.open("test.jpg"): 打开图片文件
    # preprocess(): CLIP特定的图片预处理（包括resize、裁剪、归一化等）
    # unsqueeze(0): 增加批次维度，从[C,H,W]变成[1,C,H,W]
    # to(device): 将数据移动到指定设备（CPU或GPU）
    '''
    image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)

    # 2) 准备文本（你可以改成自己的）
    # texts = [
    #     "a photo of a Chinese soldier",
    #     "a photo of a cat",
    #     "a photo of an ocean",
    #     "a photo of a hat"
    # ]

    
    #可以换成更难更高级的测试：
    # 添加更多相关的文本描述
#     texts = [
#     "A Chinese soldier in uniform",  # 更具体
#     "A person wearing a hat",
#     "Ocean waves and sea", 
#     "A cat sitting on a chair",
#     "A military personnel with equipment",
#     "A beach scene with water",
#     "A domestic animal",
#     "A person in camouflage uniform"
# ]
#     texts=[
#     "A photo of a mature woman",
#     "A photo of a cute girl",
#     "A photo of a man",
#     "A photo of a cat"
# ]

    # 或者用中文描述（CLIP支持多种语言）
    texts = [
        "海军陆战队士兵",
        "陆军士兵",
        "空军士兵",
        "一只猫",
        "夜晚的城市",
        "美食"
    ]
    

    '''
    # tokenize: 将文本转换为CLIP能理解的token序列：包括分词、添加特殊token、截断/填充到固定长度等
    '''

    text_tokens = clip.tokenize(texts).to(device)

    # 3) 编码得到embedding
    '''
    torch.no_grad(): 禁用梯度计算，减少内存使用并加速推理
    # encode_image: 提取图片特征向量
    # 输入: [batch_size, 3, 224, 224]的图片
    # 输出: [batch_size, embedding_dim]的特征向量
    '''
    with torch.no_grad():
        image_features = model.encode_image(image)   #将图片映射到特征空间中
        text_features = model.encode_text(text_tokens)  #将文本（已经转为tokens）映射到特征空间中

        # 4) 归一化后做 cosine similarity
        '''
        # 将特征向量归一化为单位向量，方便计算余弦相似度
        # norm(dim=-1): 在最后一个维度（特征维度）上计算L2范数
        # keepdim=True: 保持维度，方便除法广播
        image_features: [1, embedding_dim]
        text_features: [num_texts, embedding_dim]
        '''

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        '''
        # 计算余弦相似度
        # @: 矩阵乘法运算符
        # image_features: [1, embedding_dim]
        # text_features.T: [embedding_dim, num_texts]
        # sims: [1, num_texts]
        # squeeze(0): 去掉批次维度，变成[num_texts]
        '''
        sims = (image_features @ text_features.T).squeeze(0)  # (num_texts,)
        sims = sims.cpu().numpy()  # 这是个安全的写法，numpy一定得在CPU上运行，这里本身就在CPU上没关系，但是如果本身torch在GPU上一定要先搞到CPU上

    # 5) 输出排序结果
    '''
    # zip(texts, sims): 将文本和对应的相似度分数配对,texts和nums都是 [num_texts]
    # key=lambda x: x[1]: 匿名函数，按相似度分数（元组的第二个元素）排序
    # reverse=True: 降序排列（从高到低）
    '''
    ranked = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)


    print("Top matches:")
    for t, s in ranked:
        print(f"{s:.4f}  {t}")

if __name__ == "__main__":
    main()
