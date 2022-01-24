import cv2
import tensorflow as tf
import numpy as np
from load_imgs import load_imgs
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, 
                    help='path to dataset',        default = 'dataset/view_light_time/pomegranate')
parser.add_argument('--type',     type=str, nargs= "*",
                    help='xfield type',            default = ['light','view','time'])
parser.add_argument('--dim',      type=int, nargs= "*",
                    help='dimension of input Xfields',   default = [3,3,3])
parser.add_argument('--factor',   type=int,
                    help='downsampling factor',      default = 6)
parser.add_argument('--nfg',      type=int,
                    help='capacity multiplier',    default = 4)
parser.add_argument('--num_n',    type=int,
                    help='number of neighbors',    default = 8)
parser.add_argument('--sigma',    type=float,
                    help='bandwidth parameter',    default = 0.1)
parser.add_argument('--br',      type=float,
                    help='baseline ratio',         default = 1)
parser.add_argument('--savepath', type=str,
                    help='saving path',      default = 'results/')
parser.add_argument('--scale',      type=int,
                    help='number of intermediate points',     default = 90)
parser.add_argument('--fps',      type=float,
                    help='output video frame rate',  default = 90)

args = parser.parse_args()


def run(interpreter, input_list):
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i,data in enumerate(input_list):
        data = data.astype(np.float32)
        interpreter.set_tensor(input_details[i]['index'], data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def run_test(args):

    flows = tf.lite.Interpreter("results/juice/flows.tflite")
    flows.allocate_tensors()
    interpolated = tf.lite.Interpreter("results/juice/interpolated.tflite")
    interpolated.allocate_tensors()

    savedir = args.savepath
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    head_tail = os.path.split(args.dataset)
    savedir = os.path.join(savedir, head_tail[1])

    if not os.path.exists(savedir):
        raise NameError('There is no directory:\n %s' % (savedir))

    if not os.path.exists(os.path.join(savedir, "rendered videos")):
        os.mkdir(os.path.join(savedir, "rendered videos"))
        print('creating directory %s' % (os.path.join(savedir, "rendered videos")))

    images,coordinates,all_pairs,h_res,w_res = load_imgs(args)
    min_ = np.min(coordinates)
    max_ = np.max(coordinates)

    precomputed_flows = []

    for i in range(len(coordinates)):
        flows_out = run(flows, [coordinates[[i],::]])
        precomputed_flows.append(flows_out[0,::])
      
    precomputed_flows = np.stack(precomputed_flows,0) 
    theta = [np.pi/args.scale*i for i in range(args.scale+1)]

    X1 = 1 - np.cos(theta);
    X2 = 1 + np.cos(theta);
    Y1 = 1 + np.sqrt(1-(X1-1)**2)
    Y2 = 1 - np.sqrt(1-(X2-1)**2)
    
    X = np.append(X1,X2)
    Y = np.append(Y1,Y2)
    X = X/2
    Y = Y/2
    
    if args.type == ['time']: 
        rendering_path = np.transpose([X*(args.dim[0]-1)])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('%s/rendered videos/rendered.mp4'%(savedir),fourcc, args.fps, (w_res,h_res))
        for id in range(len(X)):
                
                input_coord = np.array([[[rendering_path[id,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]),-1))[:args.num_n]
              
              
                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]
        

                im_out = run(interpolated,[input_Neighbors, input_coord, input_coord_N,  input_flows])
                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))

                print('\r interpolated image %d of %d'%(id+1,len(rendering_path)),end=" ")

    
        out.release()

if __name__=='__main__':

    run_test(args)
    print('\n The interpolation result is located in the \'rendered videos\' folder.')
