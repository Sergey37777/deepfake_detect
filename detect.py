import glob
from helpers import *
from models import *
import torch
import os
import matplotlib.pyplot as plt

# frame_count = 30


def detect():
    filenames = glob.glob("./uploaded_videos/*.mp4")
    filenames.sort()
    num_faces = 0
    probs = []
    for filename in filenames:
        faces_ = detect_video(filename)
        if faces_ is None:
            probs.append(0.5)
            #             break
            continue

        faces = []
        for face in faces_:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = np.rollaxis(cv2.resize(face, (150, 150)), -1, 0)
            faces.append(img)

        faces2 = []
        for face in faces_:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = np.rollaxis(cv2.resize(face, (224, 224)), -1, 0)
            faces2.append(img)

        faces = torch.from_numpy(np.array(faces)).float()
        faces2 = torch.from_numpy(np.array(faces2)).float()

        inputs = []
        #         inputs.append(faces2)
        #         normalize = [False]

        #         inputs.append(faces)
        #         inputs.append(faces)
        #         inputs.append(faces)
        #         inputs.append(faces2)
        #         inputs.append(faces2)
        #         inputs.append(faces2)
        #         inputs.append(faces2)
        #         normalize = [True,True,True,False,False,False,False]

        inputs.append(faces)
        inputs.append(faces2)
        inputs.append(faces2)
        normalize = [True, False, False]

        probs_2 = []

        # CNN
        with torch.no_grad():
            for j, model in enumerate(net):
                model.eval()
                probs_ = []
                for i, face in enumerate(inputs[j]):
                    if normalize[j]:
                        face = normalize_transform(face / 255.0)
                    out = model(face[None])
                    out = torch.sigmoid(out.squeeze())
                    probs_.append(out.item())
                probs_2.append(np.mean(np.array(probs_)))

        # LRCN
        #         faces_ = []
        #         for face in faces:
        #             face = normalize_transform(face / 255.)
        #             faces_.append(face)
        #         faces_ = torch.stack(faces_, dim=0).cuda().float()

        #         with torch.no_grad():
        #             for model in net:
        #                 model.eval()
        #                 out = model(faces_[None])
        #                 out = torch.sigmoid(out.squeeze())
        #                 probs_2.append(out.item())

        mult = 1
        for prob in probs_2:
            mult *= prob
        probs.append(mult ** (1 / float(len(net))))

        num_faces += len(faces)
        print("Faces 1: {}".format(len(faces)))
        print("Faces 2: {}".format(len(faces2)))

    #         break

    print("Number of faces detected: {}".format(num_faces))
    print("Number of videos processed: {}".format(len(probs)))

    probs = np.asarray(probs)
    probs[probs != probs] = 0.5  # Remove NaNs
    # probs = np.clip(probs, 0.01, 0.99)

    plt.hist(probs, 40)

    filenames = [os.path.basename(f) for f in filenames]

    submission = pd.DataFrame({"filename": filenames, "label": probs})
    submission.to_csv("submission.csv", index=False)
    for file in filenames:
        if os.path.exists(os.path.join("./uploaded_videos/", file)):
            os.remove(os.path.join("./uploaded_videos/", file))
        else:
            print("File not exists")
    return "{0:0.2f}".format(float(probs[0]) * 100)
