
from rental.pipeline.training_pipeline import start_training_pipeline
from rental.pipeline.batch_prediction import start_batch_prediction

file_path=r"E:\project rental\bike_requirement\bike.csv"
print(__name__)
if __name__=="__main__":
     try:
          #start_training_pipeline()
          output_file = start_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e:
          print(e)