from sklearn.pipeline import Pipeline
from app.model_arena import create_annotated_dataset
from app.prediction import run_yolo_detector
from app.prediction import model_evaluation as model_eval
from app.pytorch2onnx import darknet2_onnx
# from app.pytorch2onnx import onxx2tensorflow
# import logging
from app.model_arena import train_yolov4

pipe = Pipeline(
            [
                # ('Create_dataset',create_annotated_dataset.Create_dataset(train_ratio=0.90,valid_ratio=0.10)
                #  ),
                # ('Training',train_yolov4.Training(batch_size=16,epochs=2000,verbose=False,retrain=False)
                # ),
                # ('Run_validation_inference', run_yolo_detector.Run_validation_inference(infer_path='val.txt',
                #                                         yolo_config='yolov4_custom.cfg',overlay=True)
                #  ),
                # ('Evaluate_model ', model_eval.Evaluate_model(data_txt='val.txt',
                #                                               metadata='metadata.pickle', kpi_name='KPI_metrics.csv')
                #  ),
                ('YOLO2ONXX', darknet2_onnx.YOLO2ONXX(batch_size=1, testimg='test.jpg')
                ),
                # ('YOLOONXX2TF',onxx2tensorflow.ONXX2TF(onnx_model=None,output_dir=None,testimg='test.jpg')
                #  )
            ]

        )

