import xmltodict,json,os

#根据文件夹目录，把xml文件标记信息的一些最大最小值输出
#输出结果是一个列表，列表有四个元素，每个元素又是一个列表
#第一个列表包含 图片的最大宽度，图片的最大高度，图片的最大宽高比，图片包含的最大标记数，图片包含的标记的最大类别数
#第二个列表包含 图片的最小宽度，图片的最小高度，图片的最小宽高比，图片包含的最小标记数，图片包含的标记的最小类别数
#第三个列表包含 标记的最大宽度，标记的最大高度，标记的最大宽高比，图片宽度和标记宽度的最大比值，图片高度和标记高度的最大比值
#第四个列表包含 标记的最小宽度，标记的最小高度，标记的最小宽高比，图片宽度和标记宽度的最小比值，图片高度和标记高度的最小比值
def get_folder_info(folder_path):
    files = os.listdir(folder_path)
    for fname in files:
        file_path = os.path.join(folder_path,fname)
        if fname == files[0]:
            res = get_info_from_xml(file_path)
            result = [res[:5],res[:5],res[5],res[6]]
            # print(result)
        else:
            res = get_info_from_xml(file_path)
            result1 = [res[:5],res[:5],res[5],res[6]]
            # print(result1)
            for i in range(len(result)):
                if i == 0 or i == 2:
                    for j in range(len(result[i])):
                        if result[i][j] < result1[i][j]:
                            result[i][j] = result1[i][j]
                if i == 1 or i == 3:
                    for j in range(len(result[i])):
                        if result[i][j] > result1[i][j]:
                            result[i][j] = result1[i][j]
    print(result)

#根据xml文件的路径，获取以下信息
#图片的宽度，图片的高度，图片的宽高比，图片包含的标记数，图片包含的标记的类别数
#标记的最大宽度，标记的最大高度，标记的最大宽高比，图片宽度和标记宽度的最大比值，图片高度和标记高度的最大比值
#标记的最小宽度，标记的最小高度，标记的最小宽高比，图片宽度和标记宽度的最小比值，图片高度和标记高度的最小比值
def get_info_from_xml(file_path):
    d = trans_xml_to_dict(file_path)
    info = get_pic_info_from_dict(d)
    return info

#把XML文件转换成字典
def trans_xml_to_dict(file_path):
    with open(file_path,'r') as f:
        xml_str = f.read()
    d = xmltodict.parse(xml_str)
    j = json.dumps(d)
    r = json.loads(j)
    return r

#从字典里把文件信息提取
#图片的宽度，图片的高度，图片的宽高比，图片包含的标记数，图片包含的标记的类别数
#标记的最大宽度，标记的最大高度，标记的最大宽高比，图片宽度和标记宽度的最大比值，图片高度和标记高度的最大比值
#标记的最小宽度，标记的最小高度，标记的最小宽高比，图片宽度和标记宽度的最小比值，图片高度和标记高度的最小比值
def get_pic_info_from_dict(d):
    a = d['annotation']
    pic_size = a['size']
    objects = a['object']

    w = int(pic_size['width'])
    h = int(pic_size['height'])
    wtoh = round(w/h,2)

    if type(objects) is list:
        lable_class_set = set()
        label_number = len(objects)
        max_list = get_obj_info(objects[0],w,h)[1:]
        min_list = get_obj_info(objects[0],w,h)[1:]
        for obj in objects:
            res_list = get_obj_info(obj,w,h)
            lable_class_set.add(res_list[0])
            res_list = res_list[1:]
            for i in range(len(max_list)):
                if max_list[i]<res_list[i]:
                    max_list[i] = res_list[i]
                if min_list[i]>res_list[i]:
                    min_list[i] = res_list[i]
        lable_class_number = len(lable_class_set)
    else:
        label_number = 1
        lable_class_number = 1
        max_list = get_obj_info(objects,w,h)[1:]
        min_list = get_obj_info(objects,w,h)[1:]
    return [w,h,wtoh,label_number,lable_class_number,max_list,min_list]

#获取每个标记的信息
#标记的名字，标记的宽度，标记的高度，标记的宽高比，图片宽度和标记宽度的比值，图片高度和标记高度的比值
def get_obj_info(obj,pic_w,pic_h):
    name = obj['name']
    pts = obj['bndbox']
    w = int(pts['xmax']) - int(pts['xmin']) +1
    h = int(pts['ymax']) - int(pts['ymin']) +1 
    wtoh = round(w / h,2)
    wtow = round(pic_w/w,2)
    htoh = round(pic_h/h,2)
    return [name,w,h,wtoh,wtow,htoh]

if __name__=='__main__':
    get_folder_info(r'/home/xjtu/centernet-tf2/VOCdevkit/VOC2007/Annotations')

