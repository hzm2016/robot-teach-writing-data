import os.path as osp
import cv2
import numpy as np
from PIL import Image
import io
from cairosvg import svg2png
from tqdm import tqdm
from lxml import etree
from svgpathtools import parse_path, disvg, wsvg

def list_to_str(list):

    rt_str = ''

    for ele in list:
        rt_str += ele
        rt_str += '\n'

    return rt_str

def svg_2_img(out_path, path_list):
    """[summary]

    Args:
        svg ([type]): [description]
    """

    transform = [r'<g transform="scale(1, -1) translate(0, -900)">', r'</g>']

    # for index in range(len(path_list)):

    # img_name = osp.join(out_path, str(1)) + '.jpg'
    # svg_name = osp.join(out_path, str(1)) + '.svg'
        # paths = path_list[index]
    
    img_name = out_path
    svg_name = out_path.replace('jpg','svg')
        
    path_str = wsvg(path_list,filename=svg_name,dimensions=(1024,1024)) #,viewbox='0 0 1024 1024')
    path_str_list = path_str.split('\n')
    path_str_list.insert(2, transform[0])
    path_str_list.insert(-2, transform[1])
    path_str = list_to_str(path_str_list)
    svg2png(bytestring=path_str,write_to=img_name,background_color='white',output_width=128, output_height=128)

def svg2img(paths):   

    paths = parse_path(paths)
    # transform = [r'<g transform="scale(1, -1) translate(0, -900)">', r'</g>']    
    transform = [r'<g transform="scale(1, -1) translate(0, -900)"> ', r'</g>']    
    path_str = wsvg(paths, filename='test.svg', dimensions=(1024, 1024), stroke_widths=[1])
    path_str_list = path_str.split('\n')  
    path_str_list.insert(2, transform[0])
    path_str_list.insert(-2, transform[1])
    path_str = list_to_str(path_str_list)

    # change the fill color of the svg
    root = etree.fromstring(path_str)
    tree = etree.ElementTree(root)
    color = '#000000'
    root.attrib["fill"] = color
    root.attrib["width"] = '128'
    root.attrib["height"] = '128'

    for path in root.iter():
        path.attrib["fill"] = color

    path_str = etree.tostring(tree.getroot(), pretty_print=True).decode('utf-8')
    image_io = svg2png(bytestring=path_str,background_color='white')#,output_width=128, output_height=128)
    # io.seek(0)
    image_io = io.BytesIO(image_io)
    byteImg = np.asarray(Image.open(image_io))
    # cv2.imwrite('test.jpg',byteImg[:,:,0])
    # aa = byteImg[:,:,0]
    # aa = aa - 255
    # print(np.nonzero(aa))
    # import pdb;pdb.set_trace()  
    return byteImg[:,:,0]


def _parse_strokes(strokes):
    """[summary]

    Args:
        strokes ([type]): [description]
    """

    path_list = []

    for stroke in strokes:

        path = parse_path(stroke)
        path_list.append(path)

    return path_list

def main():
    """[summary]
    """

    input_file = './tools/src/graphics.txt'
    output_dir = './imgs/D'

    input_lines = open(input_file,'r').readlines()

    for idx, line in enumerate(tqdm(input_lines)):

        char_info = eval(line)

        strokes = char_info['strokes']
        medians = char_info['medians']
        char = char_info['character']

        out_path = osp.join(output_dir, str(idx)) + '.jpg'
        # os.makedirs(out_path, exist_ok = True)
        svg = _parse_strokes(strokes)
        svg_2_img(out_path, svg)

if __name__ == '__main__':
    main()