# if __name__ == "__main__":
#     raise ChildProcessError
import clearml
import os
#import comet_ml
#from comet_ml import API
from roboflow import Roboflow
from IPython import display
import shutil
from functools import partial
import numpy as np
import dotenv

#display.clear_output()

import ultralytics


import torch

GPUS_AVALIBLE = torch.cuda.device_count()
HOME = os.getcwd()

gpu_array = [x for x in range(0,GPUS_AVALIBLE)]
gpu_array = ",".join(str(e) for e in gpu_array)
#%env CUDA_VISIBLE_DEVICES={gpu_array}
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_array
#%env OMP_NUM_THREADS=16
os.environ["OMP_NUM_THREADS"]='1'

dotenv.load_dotenv(".env")


ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


class YOLO_Trainer():
    def __init__(self,dataset_version='latest',  model='yolov8n.pt', task="detect", workspace_id="trashclassification-tayqe", project_id="trash-xvysc", comet_workspace='eilliw', classify_workspace_id="trashclassification-tayqe", classify_project_id="trash-classification-vky3a", clearml_project_name="trash-classification") -> None:

        self.dataset_version = dataset_version
        self.workspace_id = workspace_id
        self.project_id = project_id

        self.classify_workspace_id = classify_workspace_id
        self.classify_project_id = classify_project_id

        self.clearml_project_name = clearml_project_name

        self.task = task

        self.model_name = model
        self.clean_model_name =  self.model_name.removesuffix(".pt")
        self.model: YOLO = YOLO(model)
        
        self.comet_workspace = comet_workspace

        self.dataset = self._get_dataset(self.workspace_id, self.project_id, task)
        self.fix_dataset()

    def train(self, project, exist_ok=False, name = None, dummy=False, task='detect', plots = True, imgsz = 640, save_json = True, epochs=100, save_period=5, workers=16, bs=64,  export=None, cache=False):
        if name != None:
            try:
                os.makedirs(project+"//"+name)
            except FileExistsError:
                print(project+"//"+name+" Already exists")
        
        print(ultralytics.checks())
        print(f"# of GPU's avalible - {GPUS_AVALIBLE}")
        #self._setup_comet(project)
       # self.experiment.add_tag(self.clean_model_name)
        #self.experiment.add_tag(f"Dataset-V:{self.dataset.version}")
        #self.experiment.add_tag(f"Task:  {task}")

        #log_model_status = self.experiment.log_model(self.clean_model_name, f"{project}//{name}//weights", file_name=f"{project}//{name}//weights//best.pt")

        #data_dir = f"{self.dataset.location}/data.yaml"
        self._setup_clearml()
        if task =="classify":
            data_dir = f"{self.dataset.location}"
        if not dummy:
            print("visible devices",os.getenv("CUDA_VISIBLE_DEVICES"))
            #torch.distributed.init_process_group()

            cuda_int_list = [x for x in range(0,GPUS_AVALIBLE)]
            print(f"int list of cuda devices: {cuda_int_list}")

            self.results = self.model.train( device=cuda_int_list, task = task, project=project, data = data_dir, name=name, exist_ok=exist_ok, imgsz=imgsz,plots=plots, save_period=save_period, save_json= save_json, workers=workers, batch=bs, epochs=epochs, cache=cache)
        #self.experiment.register_model(self.clean_model_name, version=self._get_model_version(self.clean_model_name, self.dataset_version, 1))
        #self.experiment.end()
        if export is not None:
            self.export(export, name=self.clean_model_name)
        return self.results
    def export(self, format, location="exports", name="best"):
        
        exported_model_path = self.model.export(format=format)
        desired_path = f"{location}//{name}//{format}"
        try:
            os.makedirs(desired_path)
        except FileExistsError:
            pass
        shutil.move(exported_model_path, desired_path)
    def fix_dataset(self):
        try:
            os.rename(self.dataset.location+"/valid",self.dataset.location+"/val")
        except:
            pass
        return
    def _get_dataset(self, workspace_id, project_id, task):
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        
        dataset_type = "yolov8"

        if task =="classify":
            dataset_type="folder"
            project = rf.workspace(self.classify_workspace_id).project(self.classify_project_id)

        elif task=="detect":
            project = rf.workspace(workspace_id).project(project_id)
            dataset_type = "yolov8"
        if self.dataset_version == 'latest':
                
            project_versions = len(project.versions())
            dataset_abs_path = os.path.abspath(f"{os.getcwd()}/datasets/{self.classify_project_id}/{project_versions}/{dataset_type}")
            
            dataset = project.version(project_versions).download(dataset_type,overwrite=False, location=dataset_abs_path)
            return dataset
        else:
            dataset_abs_path = os.path.abspath(f"{os.getcwd()}/datasets/trash-{self.project_version}/{dataset_type}")
            dataset = project.version(self.project_version).download(dataset_type,overwrite=False, location=dataset_abs_path)

            return dataset
    def _get_classify_dataset(self):
        pass

    def _setup_clearml(self):
        from clearml import Task
        self.task = Task.init(project_name=self.clearml_project_name, reuse_last_task_id=False)

    def _setup_comet(self, name):
        self.experiment = comet_ml.Experiment(project_name=name)
    def _get_model_version(self, model, dataset_version, alg_and_images_ver):
        api = API()
        sample_version = [str(alg_and_images_ver),str(dataset_version),'0']
        try:
            versions: list[str] = api.get_registry_model_versions(self.comet_workspace, model)
        except:
            return sample_version
        #formated_versions = map(partial(versions)
        formated_versions = [ver.split('.') for ver in versions]
        # np_vers = np.array(formated_versions)
        # np.unr
        try:
            match_ds_ver = [x for x in formated_versions if x[1]==str(dataset_version)]
            max_2_vers = max(match_ds_ver, key= lambda x: x[2])
            if not isinstance(any(max_2_vers), list):
                sample_version[2] = str(int(max_2_vers[2])+1)
            else:
                sample_version[2] = str(int(max_2_vers[0][2])+1)
            return '.'.split(sample_version)
        except:
            return '.'.split(sample_version)

#print(ultralytics.checks())


# print(GPUS_AVALIBLE)

# print(HOME)

# rf = Roboflow(api_key="vFBCQUPZEIwkQ9xEqkDu")
# project = rf.workspace("trashclassification-tayqe").project("trash-xvysc")
# project_versions = project.get_version_information()
# version_max = project_versions[0]['id'][-1]
# print(f"dataset_version: {version_max}")
# dataset = project.version(version_max).download("yolov8",overwrite=False, location=f"{HOME}//datasets//trash-{version_max}")
# dataset_version = dataset.version

#%env COMET_API_KEY=PlUu2e0X1ob9KsjX9edXGbIep


# experiment = comet_ml.Experiment(
#     project_name="trash-classification"
# )






from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).name)
print("hello")


#model_path = "runs\detect\\train36\\weights\\best.pt"

#model = YOLO(model_path, task='detect')  # load a pretrained model (recommended for training)

#
#name=  "trash-classifier_v1"
#project_directory = f"{HOME}/runs"
# model_name  = "yolov8m.pt"
# bs = 30
# epochs = 75
# if GPUS_AVALIBLE >1:
#     experiment.add_tag(model_name)
#     experiment.add_tag(f"Dataset-V:{dataset.version}")
#     model = YOLO(model_name)
#     bs = 32
#     epochs = 100
#     workers = 16
#     save_period = 5
#     bs=64
#     # !yolo task=detect mode=train model={model_name} data={dataset.location}/data.yaml epochs=100 batch={bs} imgsz=640 plots=True device={CUDA_VISIBLE_DEVICES} save_json=True project=trash-classification workers={workers} save_period={save_period}
    
#     #results = model.train(data=f"{dataset.location}//data.yaml", epochs=epochs, plots=True,  imgsz=640, batch=bs,device =[x for x in range(0,GPUS_AVALIBLE)],save_json=True, save_period=5,project='trash-classification',workers=workers)
#     print(os.getcwd())
#     model.load("trash-classification/train14/weights/best.pt")
#     model.export(format="onnx")
# elif GPUS_AVALIBLE==1:
#     model = YOLO('yolov8s.pt')
#     bs = 30
#     epochs = 50
#     results = model.train(data=f"{dataset.location}//data.yaml", epochs=epochs, plots=True,  imgsz=640, batch=bs)

# # Use the model
# #model.train(data=f"{dataset.location}/data.yaml", epochs=25, plots=True, task='detect',  imgsz=800, project=project_directory, batch=bs)  # train the model
# #model.train(data=f"{dataset.location}//data.yaml", epochs=epochs, plots=True,  imgsz=640, batch=bs)
# #metrics = model.val()  # evaluate model performance on the validation set
# #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#   # export the model to ONNX format
# #path = model.export(format='simplify')
# experiment.end()
