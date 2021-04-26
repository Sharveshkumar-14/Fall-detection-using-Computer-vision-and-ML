import os
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from flask import Flask, render_template, url_for, redirect, session, Response ,request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import json, random

import pandas as pd
# importing libraries for ML model

import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from torchvision import transforms

# This grabs our directory
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder='template')
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

app.config['SECRET_KEY'] = 'mysecretkey'

# sending mail to the user
app.config['TESTING'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'sharveshkumarcena2000@gmail.com'
app.config['MAIL_PASSWORD'] = '123'
app.config['MAIL_DEFAULT_SENDER'] = 'sharveshkumarcena2000@gmail.com'
app.config['MAIL_MAX_EMAILS'] = None
app.config['MAIL_ASCII_ATTACHMENTS']= False

mail = Mail(app)
# Connects our Flask App to our Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class video_actions(db.Model):

    id = db.Column(db.Integer,primary_key = True)
    actname = db.Column(db.Text)

    # def __init__(self, actname):
    #     self.actname = actname

    # def __repr__(self, actname):
    #     return "<actname: {}>".format(self.actname)


class addentry(db.Model):

    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.Text)
    mail = db.Column(db.Text)
    
    # def __repr__(self, name, mail):
    #     return "<Name: {}>".format(self.name)


class configForm(FlaskForm):
    videosource= StringField("Video Source")
    outputdirectory=StringField("Output Directory")
    submit=SubmitField("Start")

@app.route('/')
def index():
    return render_template('base.html')


# functions used for ML model

def preproc(image):
    """preprocess function for CameraLoader.
    """
    resize_fn = ResizePadding(384, 384)
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

# global counter 
global counter 
counter = False
#actionnames = []
def gen(my_var):
    action_dict = {'Standing' : 0, 'Walking': 0, 'Sitting': 0, 'Lying Down': 0,
               'Stand up':0, 'Sit down': 0, 'Fall Down': 0}
    actionnames = []
    #source='C:/Users/Sharvesh Kumar/Downloads/fall.mp4'
    source = my_var
    #print(my_var)
    #print(my_var)
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='True',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device
    # device= torch.args.device("cpu")

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    def preproc(image):
        """preprocess function for CameraLoader.
        """
        # resize_fn = ResizePadding(384, 384)
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = True
    
    if args.save_out != '':
        video = cv2.VideoCapture(source)
        frame_width = int(video.get(3)) 
        frame_height = int(video.get(4)) 
        size = (frame_width, frame_height)
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')


    fps_time = 0
    f = 0
    cnt = 0
    while cam.grabbed():
        f += 1
        # Single Frame
        frame = cam.getitem()
        image = frame.copy()

        # print(frame.shape)

        # # Skip 10 frames
        # ret, frame = video.read(cnt)
        # frame = np.array(frame)
        # print(frame.shape)
        # image = frame
        # cnt += 5
        # video.set(1,cnt)
         
        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)   
                    counter = True
                    #actionslist.append(action_name)                 
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
                # print(f,action_name)
                actionnames.append([action_name])
                action_dict[action_name] += 1
                print(action_dict)
                dataframe = pd.DataFrame(action_dict, index=[0])
                dataframe.to_csv('fall.csv', index=False)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]

        cv2.imwrite('predictions/' + str(f) + '.jpg', frame)
        raw_frame = frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        fps_time = time.time()

        if f == 1:
          size = (raw_frame.shape[0],raw_frame.shape[1])
          # print(size)
          writer = cv2.VideoWriter('output_fall.avi',  
                         codec, 
                         20, size)

        
        if outvid:
            writer.write(raw_frame)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # Clear resource.
    # cam.stop()
    # an = None
    # for i in actionnames:
    #     an = video_actions(actname = i)
    #     db.session.add(an)
    #     db.session.commit()
    if outvid:
        writer.release()
    # cv2.destroyAllWindows()
    # msg = Message('Heyy u got it', recipients=['shar17308.cs@rmkec.ac.in'])
    # mail.send(msg)
    dataframe = pd.DataFrame(action_dict, index=[0])
    dataframe.to_csv('fall.csv', index=False)

def gen_test():
    cap = cv2.VideoCapture('output_fall.avi')

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5 ,fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break 
@app.route('/bv')
def bv():

    return Response(gen_test(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/browse')
def browse():

    return render_template('browse.html')


@app.route('/livestream')
def live_stream():
    """Video streaming route. Put this in the src attribute of an img tag."""
    my_var = session.get('my_var', None)

    # msg = Message('Heyy u got it', recipients=['shar17308.cs@rmkec.ac.in'])
    # mail.send(msg)
 
    return Response(gen(my_var),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/live', methods=['GET','POST'])
def live():

    
    my_var = session.get('my_var', None)
    my_mail = session.get('my_mail', None)
    my_actions = session.get('my_actions', None)


    # msg = Message('Heyy u got it', recipients=['shar17308.cs@rmkec.ac.in'])
    # mail.send(msg)


    return render_template('live_stream.html')



@app.route('/Home', methods=['GET','POST'])
def Home():
    videosource = False
    outputdirectory = False
    form = configForm()
    if form.validate_on_submit():
        videosource = form.videosource.data
        # aname = video_actions(actname = videosource)
        # db.session.add(aname)
        # db.session.commit()
        outputdirectory = form.outputdirectory.data
        session['my_var'] = videosource
        return redirect(url_for('live'))
        # form.videosource.data = ''
        # form.outputdirectory.data= ''
    return render_template ('home.html',form=form, videosource=videosource, outputdirectory=outputdirectory)

@app.route('/emergency',methods=['GET','POST'])
def emergencycontact():
    users = None
    if request.form:
        user= addentry(name=request.form.get("name"),mail=request.form.get("mail"))
        session['my_mail'] = user.mail
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('emergencycontact'))

    users = addentry.query.all()
    return render_template('emergency.html',users= users)

@app.route("/delete", methods=["GET","POST"])
def delete():
    mail = request.form.get("mail")
    user = addentry.query.filter_by(mail = mail).first()
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for('emergencycontact'))


@app.route("/analyse", methods = ["GET","POST"])
def analyse():
    result_dict = pd.read_csv('fall.csv', header=None)
    action_names, action_counts = result_dict.values.tolist()[0], result_dict.values.tolist()[1]
    print(action_names,action_counts)
    
    ind = action_names.index('Fall Down')
    #print(ind,action_counts[ind])
    if int(action_counts[ind]) > 0 :
        msg = Message('The person has fallen', recipients=['shar17308.cs@rmkec.ac.in'])
        mail.send(msg)
        flash('The person has fallen')
        
    return render_template("analyse.html", actionnames = action_names, actcount = action_counts)

@app.route("/xdata", methods = ["GET","POST"])
def xdata():
    result_dict = pd.read_csv('fall.csv', header=None)
    action_names, action_counts = result_dict.values.tolist()[0], result_dict.values.tolist()[1]
    
    data ={
        "xdata" : action_names
    }
    
    return data

@app.route("/ydata", methods = ["GET","POST"])
def ydata():
    result_dict = pd.read_csv('fall.csv', header=None)
    action_names, action_counts = result_dict.values.tolist()[0], result_dict.values.tolist()[1]
    data ={
        "ydata" : action_counts
    }
    
    return data



@app.route("/about",  methods=["GET","POST"])
def about():

    my_setlist = session.get('my_setlist', None)
    my_actlist = session.get('my_actcount', None)
    # print(my_setlist)
    # print(my_actlist)


    return render_template('about.html')

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
