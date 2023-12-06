import fiftyone as fo
import dotenv
#from triton_client.inference import InferenceClient

if __name__ == "__main__":
    #load api keys and other environmental variables
    dotenv.load_dotenv(".env")
    from tools.dataset import PiCamDataset
    #create triton server connection

    #loading dataset from class
    picamdataset = PiCamDataset()
    picamdataset.download_latest()
    dataset_location = picamdataset.dataset.location

    #pointing to fiftyone dataset
    dataset_type = fo.types.ImageClassificationDirectoryTree
    fo_dataset = fo.Dataset.from_dir(
        dataset_location,
        dataset_type=dataset_type,
        name=picamdataset.dataset.name)
    print(f"fiftyone dataset created: {fo_dataset}")
    session = fo.launch_app(fo_dataset)
    session.wait()