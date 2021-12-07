FROM continuumio/miniconda3
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive

# download dependencies for opencv
RUN apt-get update 
RUN apt-get install 'libsm6' 'libxext6' 'libgl1-mesa-glx'  -y

# Create the environment
COPY environment.yml /app
RUN conda env create -f environment.yml

# Activate the environment and add in PATH :
RUN echo "source activate ocr_model" > ~/.bashrc
ENV PATH /opt/conda/envs/ocr_model/bin:$PATH

# Make RUN commands use the  new environment
SHELL ["/bin/bash", "--login", "-c"]

# Copy all src files in app
COPY . /app
# Make EASYOCR directory
RUN mkdir -p ~/.EasyOCR/model
# Copy all easyocr model files in easyocr dir 
RUN mv /app/src/model/arabic.pth ~/.EasyOCR/model/
RUN mv /app/src/model/craft_mlt_25k.pth ~/.EasyOCR/model/
RUN mv /app/src/model/english_g2.pth ~/.EasyOCR/model/

# Expose Docker port  
EXPOSE 80
# Create Volume
VOLUME [ "/app/artifacts" ]
# Run the application:

CMD ["python","app.py"]
