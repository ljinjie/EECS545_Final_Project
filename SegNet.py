import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SegNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=31):
        super(SegNet, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        # Encoder structure (5 stages)
        # Stage 1
        self.EnConvS1_1 = nn.Conv2d(self.in_channel, 64, kernel_size=(3, 3), padding=1)
        self.EnBNS1_1 = nn.BatchNorm2d(64)
        self.EnConvS1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.EnBNS1_2 = nn.BatchNorm2d(64)
        # Stage 2
        self.EnConvS2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.EnBNS2_1 = nn.BatchNorm2d(128)
        self.EnConvS2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.EnBNS2_2 = nn.BatchNorm2d(128)
        # Stage 3
        self.EnConvS3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.EnBNS3_1 = nn.BatchNorm2d(256)
        self.EnConvS3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.EnBNS3_2 = nn.BatchNorm2d(256)
        self.EnConvS3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.EnBNS3_3 = nn.BatchNorm2d(256)
        # Stage 4
        self.EnConvS4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS4_1 = nn.BatchNorm2d(512)
        self.EnConvS4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS4_2 = nn.BatchNorm2d(512)
        self.EnConvS4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS4_3 = nn.BatchNorm2d(512)
        # Stage 5
        self.EnConvS5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS5_1 = nn.BatchNorm2d(512)
        self.EnConvS5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS5_2 = nn.BatchNorm2d(512)
        self.EnConvS5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.EnBNS5_3 = nn.BatchNorm2d(512)
        # Encoder max-pooling layer
        self.EnMP = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder structure (5 stages)
        # Stage 5
        self.DeConvS5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.DeBNS5_3 = nn.BatchNorm2d(512)
        self.DeConvS5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.DeBNS5_2 = nn.BatchNorm2d(512)
        self.DeConvS5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.DeBNS5_1 = nn.BatchNorm2d(512)
        # Stage 4
        self.DeConvS4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.DeBNS4_3 = nn.BatchNorm2d(512)
        self.DeConvS4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.DeBNS4_2 = nn.BatchNorm2d(512)
        self.DeConvS4_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1)
        self.DeBNS4_1 = nn.BatchNorm2d(256)
        # Stage 3
        self.DeConvS3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.DeBNS3_3 = nn.BatchNorm2d(256)
        self.DeConvS3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.DeBNS3_2 = nn.BatchNorm2d(256)
        self.DeConvS3_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.DeBNS3_1 = nn.BatchNorm2d(128)
        # Stage 2
        self.DeConvS2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.DeBNS2_2 = nn.BatchNorm2d(128)
        self.DeConvS2_1 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.DeBNS2_1 = nn.BatchNorm2d(64)
        # Stage 1
        self.DeConvS1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.DeBNS1_2 = nn.BatchNorm2d(64)
        self.DeConvS1_1 = nn.Conv2d(64, self.out_channel, kernel_size=(3, 3), padding=1)
        self.DeBNS1_1 = nn.BatchNorm2d(self.out_channel)
        # Decoder upsampling layer
        self.DeMP = nn.MaxUnpool2d(2, stride=2)

        self.init_weights()

    def forward(self, x):
        # Encoder structure (5 stages)
        # Stage 1
        x = F.relu(self.EnBNS1_1(self.EnConvS1_1(x)))
        x = F.relu(self.EnBNS1_2(self.EnConvS1_2(x)))
        x, EnS1_idx = self.EnMP(x)
        EnS1_size = x.size()

        # Stage 2
        x = F.relu(self.EnBNS2_1(self.EnConvS2_1(x)))
        x = F.relu(self.EnBNS2_2(self.EnConvS2_2(x)))
        x, EnS2_idx = self.EnMP(x)
        EnS2_size = x.size()

        # Stage 3
        x = F.relu(self.EnBNS3_1(self.EnConvS3_1(x)))
        x = F.relu(self.EnBNS3_2(self.EnConvS3_2(x)))
        x = F.relu(self.EnBNS3_3(self.EnConvS3_3(x)))
        x, EnS3_idx = self.EnMP(x)
        EnS3_size = x.size()

        # Stage 4
        x = F.relu(self.EnBNS4_1(self.EnConvS4_1(x)))
        x = F.relu(self.EnBNS4_2(self.EnConvS4_2(x)))
        x = F.relu(self.EnBNS4_3(self.EnConvS4_3(x)))
        x, EnS4_idx = self.EnMP(x)
        EnS4_size = x.size()

        # Stage 5
        x = F.relu(self.EnBNS5_1(self.EnConvS5_1(x)))
        x = F.relu(self.EnBNS5_2(self.EnConvS5_2(x)))
        x = F.relu(self.EnBNS5_3(self.EnConvS5_3(x)))
        x, EnS5_idx = self.EnMP(x)

        # Decoder structure
        # Stage 5
        x = self.DeMP(x, EnS5_idx, output_size=EnS4_size)
        x = F.relu(self.DeBNS5_3(self.DeConvS5_3(x)))
        x = F.relu(self.DeBNS5_2(self.DeConvS5_2(x)))
        x = F.relu(self.DeBNS5_1(self.DeConvS5_1(x)))

        # Stage 4
        x = self.DeMP(x, EnS4_idx, output_size=EnS3_size)
        x = F.relu(self.DeBNS4_3(self.DeConvS4_3(x)))
        x = F.relu(self.DeBNS4_2(self.DeConvS4_2(x)))
        x = F.relu(self.DeBNS4_1(self.DeConvS4_1(x)))

        # Stage 3
        x = self.DeMP(x, EnS3_idx, output_size=EnS2_size)
        x = F.relu(self.DeBNS3_3(self.DeConvS3_3(x)))
        x = F.relu(self.DeBNS3_2(self.DeConvS3_2(x)))
        x = F.relu(self.DeBNS3_1(self.DeConvS3_1(x)))

        # Stage 2
        x = self.DeMP(x, EnS2_idx, output_size=EnS1_size)
        x = F.relu(self.DeBNS2_2(self.DeConvS2_2(x)))
        x = F.relu(self.DeBNS2_1(self.DeConvS2_1(x)))

        # Stage 1
        x = self.DeMP(x, EnS1_idx)
        x = F.relu(self.DeBNS1_2(self.DeConvS1_2(x)))
        x = self.DeConvS1_1(x)

        # SoftMax classifier
        # x = F.softmax(x, dim=-1)

        return x

    def init_weights(self):
        # initialize encoder with pretrained VGG16 weights
        vgg = models.vgg16_bn(pretrained=True)

        self.EnConvS1_1.weight.data.copy_(vgg.features[0].weight.data)
        self.EnConvS1_1.bias.data.copy_(vgg.features[0].bias.data)
        self.EnBNS1_1.weight.data.copy_(vgg.features[1].weight.data)
        self.EnBNS1_1.bias.data.copy_(vgg.features[1].bias.data)
        self.EnConvS1_2.weight.data.copy_(vgg.features[3].weight.data)
        self.EnConvS1_2.bias.data.copy_(vgg.features[3].bias.data)
        self.EnBNS1_2.weight.data.copy_(vgg.features[4].weight.data)
        self.EnBNS1_2.bias.data.copy_(vgg.features[4].bias.data)

        self.EnConvS2_1.weight.data.copy_(vgg.features[7].weight.data)
        self.EnConvS2_1.bias.data.copy_(vgg.features[7].bias.data)
        self.EnBNS2_1.weight.data.copy_(vgg.features[8].weight.data)
        self.EnBNS2_1.bias.data.copy_(vgg.features[8].bias.data)
        self.EnConvS2_2.weight.data.copy_(vgg.features[10].weight.data)
        self.EnConvS2_2.bias.data.copy_(vgg.features[10].bias.data)
        self.EnBNS2_2.weight.data.copy_(vgg.features[11].weight.data)
        self.EnBNS2_2.bias.data.copy_(vgg.features[11].bias.data)

        self.EnConvS3_1.weight.data.copy_(vgg.features[14].weight.data)
        self.EnConvS3_1.bias.data.copy_(vgg.features[14].bias.data)
        self.EnBNS3_1.weight.data.copy_(vgg.features[15].weight.data)
        self.EnBNS3_1.bias.data.copy_(vgg.features[15].bias.data)
        self.EnConvS3_2.weight.data.copy_(vgg.features[17].weight.data)
        self.EnConvS3_2.bias.data.copy_(vgg.features[17].bias.data)
        self.EnBNS3_2.weight.data.copy_(vgg.features[18].weight.data)
        self.EnBNS3_2.bias.data.copy_(vgg.features[18].bias.data)
        self.EnConvS3_3.weight.data.copy_(vgg.features[20].weight.data)
        self.EnConvS3_3.bias.data.copy_(vgg.features[20].bias.data)
        self.EnBNS3_3.weight.data.copy_(vgg.features[21].weight.data)
        self.EnBNS3_3.bias.data.copy_(vgg.features[21].bias.data)

        self.EnConvS4_1.weight.data.copy_(vgg.features[24].weight.data)
        self.EnConvS4_1.bias.data.copy_(vgg.features[24].bias.data)
        self.EnBNS4_1.weight.data.copy_(vgg.features[25].weight.data)
        self.EnBNS4_1.bias.data.copy_(vgg.features[25].bias.data)
        self.EnConvS4_2.weight.data.copy_(vgg.features[27].weight.data)
        self.EnConvS4_2.bias.data.copy_(vgg.features[27].bias.data)
        self.EnBNS4_2.weight.data.copy_(vgg.features[28].weight.data)
        self.EnBNS4_2.bias.data.copy_(vgg.features[28].bias.data)
        self.EnConvS4_3.weight.data.copy_(vgg.features[30].weight.data)
        self.EnConvS4_3.bias.data.copy_(vgg.features[30].bias.data)
        self.EnBNS4_3.weight.data.copy_(vgg.features[31].weight.data)
        self.EnBNS4_3.bias.data.copy_(vgg.features[31].bias.data)

        self.EnConvS5_1.weight.data.copy_(vgg.features[34].weight.data)
        self.EnConvS5_1.bias.data.copy_(vgg.features[34].bias.data)
        self.EnBNS5_1.weight.data.copy_(vgg.features[35].weight.data)
        self.EnBNS5_1.bias.data.copy_(vgg.features[35].bias.data)
        self.EnConvS5_2.weight.data.copy_(vgg.features[37].weight.data)
        self.EnConvS5_2.bias.data.copy_(vgg.features[37].bias.data)
        self.EnBNS5_2.weight.data.copy_(vgg.features[38].weight.data)
        self.EnBNS5_2.bias.data.copy_(vgg.features[38].bias.data)
        self.EnConvS5_3.weight.data.copy_(vgg.features[40].weight.data)
        self.EnConvS5_3.bias.data.copy_(vgg.features[40].bias.data)
        self.EnBNS5_3.weight.data.copy_(vgg.features[41].weight.data)
        self.EnBNS5_3.bias.data.copy_(vgg.features[41].bias.data)
