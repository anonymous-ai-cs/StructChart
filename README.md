<div align="center">
<h1>StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding<br></h1>
</div>

Charts are common in literature across different scientific fields, conveying rich information easily accessible to readers. Current chart-related tasks focus on either chart perception which refers to extracting information from the visual charts, or performing reasoning given the extracted data, e.g. in a tabular form. In this paper, we aim to establish a unified and label-efficient learning paradigm for joint perception and reasoning tasks, which can be generally applicable to different downstream tasks, beyond the question-answering task as specifically studied in peer works. Specifically, StructChart first reformulates the chart information from the popular tubular form (specifically linearized CSV) to the proposed Structured Triplet Representations (STR), which is more friendly for reducing the task gap between chart perception and reasoning due to the employed structured information extraction for charts. We then propose a Structuring Chart-oriented Representation Metric (SCRM) to quantitatively evaluate the performance for the chart perception task. To enrich the dataset for training, we further explore the possibility of leveraging the Large Language Model (LLM), enhancing the chart diversity in terms of both chart visual style and its statistical information. Extensive experiments are conducted on various chart-related tasks, demonstrating the effectiveness and promising potential for a unified chart perception-reasoning paradigm to push the frontier of chart understanding. 

 
## Installation
a. Download the pre-trained model pix2struct
[Hugging Face](https://huggingface.co/google/pix2struct-base)

b. Install the dependent libraries as follows:

  ```shell
    pip install -r requirements.txt 
  ```

## Setting up Data
You have to first preprocess the data, we give a template file in `tools/data_preprocess/data_preprocess_chartQA.py`
Then you should specify the value of `--data_root` in `--config`. This should be the absolute path of the datasets.

The following datasets are used in our paper: 
- ChartQA \[[Dataset Page](https://github.com/vis-nlp/ChartQA)\]
- PlotQA \[[Dataset Page](https://github.com/NiteshMethani/PlotQA)\]
- Chart2Text \[[Dataset Page](https://github.com/JasonObeid/Chart2Text)\]
- SimChart9K \[[Download](https://drive.google.com/file/d/1kqov-01SfVT4hgeXXNRQPS5NyyuSqbih/view?usp=sharing)\]

## Train and Test
* Train using multi-GPU
```shell script
sh scripts/dist_train.sh 8 \
--config ./cfgs/chartQA/structchart_base.yaml
--VAL_PER_EPOCH 0
```

* Test using multi-GPU
```shell script
sh scripts/dist_test.sh 8 \
--config ./cfgs/chartQA/structchart_base.yaml \
--ckpt ${CHECKPOINT_PATH} \
--num_token 1280 \
--criterion csv_metric 
```
## Main Results
Here, we present the performance of StrcutChart on ChartQA val set. All the experiments are evaluated by our proposed Structuring Chart-oriented Representation Metric (SCRM).

| Train set                 | mPrecison (strict) | mPrecison (slight) | mPrecison (high) | ckpts |
|---------------------------|--------------------|--------------------|------------------|-------|
| ChartQA                   | 0.6770             | 0.7792             | 0.8274           |\[[Download]()\]|
| ChartQA+PlotQA+Chart2Text | 0.7017             | 0.8227             | 0.8591           |\[[Download]()\]|
| ChartQA 0.2+SimChart9K    | 0.6465             | 0.7787             | 0.8206           |\[[Download]()\]|
| ChartQA 0.5+SimChart9K    | 0.6902             | 0.8015             | 0.8380           |\[[Download]()\]|
| ChartQA+SimChart9K        | 0.7116             | 0.8182             | 0.8527           |\[[Download]()\]|
