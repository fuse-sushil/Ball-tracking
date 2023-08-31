import cv2
import os


class frame_extrator:
    def __init__(self,path,frame_skip):
        self.path=path
        self.frame_skip=frame_skip
    def make_folder(self,fol_name):
        try:
            os.mkdir(fol_name)
        except:
            print('Folder Already Created')

    def generate_frame(self, start_frame_num=0):
        fol_name=(self.path.split('/')[-1]).split('.')[0]
        self.make_folder(fol_name)
        cap=cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()

            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if not ret:
                break
            
            if frame_no < start_frame_num:
                continue
            else:
                if frame_no % self.frame_skip != 0:
                    continue
                else:
                    cv2.imshow('output',frame)
                    key=cv2.waitKey(0)
                    if key== ord('q'):
                        break
                    elif key==ord('s') or key == ord('S'):
                        image_path = os.path.join(fol_name, f"{fol_name}_frame_{frame_no}.jpg")
                        cv2.imwrite(image_path, frame)
                        print(f"Saved image: {image_path}")
                    else:
                        continue

        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    '''
    Put path of the video and frame skip  then rest is carried out automaticlly. Press 'q' to terminate the program
    Press 's' or 'S' to save the particular file, Any other key will skip 10 frames

    '''

    video_path= "videos/unseen-8.mp4" 
    frame_skip=10
    f_e=frame_extrator(video_path, frame_skip)
    f_e.generate_frame(start_frame_num=1)