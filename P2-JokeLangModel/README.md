# Files being used

|-- tensorflow-char-rnn/
|   |-- data/
|   |   |-- stupidstuff.txt      ## joke dataset for training (originally in json format)
|   |-- train.py                 ## training script
|   |-- sample.py                ## sampling script
|   |-- output/                  ## the default output path , with the example model trained using joke dataset

# Quick start for training

`python train.py --data_file=data/stupidstuff.txt --num_epochs=10`

The default setting of num_epochs is 50, feel free to test it.

# Quick start for sampling

`python sample.py --init_dir=output/ --start_text="{ "body":" --length=1000`

Because the training dataset isn't cleaned, to match the format for a paragraph, here sets the start_text to "{ "body":".
Length is the length of sampled sequence (# of characters), which is set to 1000 to include at least one "full joke output". Feel free to test different numbers.