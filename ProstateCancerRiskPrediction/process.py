# import SimpleITK as sitk
from pathlib import Path
import json
import random

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from monai import transforms, networks
import torch
from torch.nn.functional import softmax
from EfficientNetMultimodal import EfficientNetMultimodal
from bimcv_aikit.models.classification import SwinViTMultimodal_v3


class Prostatecancerriskprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # path to image file
        self.image_input_dir = "/input/images/axial-t2-prostate-mri/"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]

        # load clinical information
        # dictionary with patient_age and psa information
        with open("/input/psa-and-age.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.risk_score_output_file = Path("/output/prostate-cancer-risk-score.json")
        self.risk_score_likelihood_output_file = Path(
            "/output/prostate-cancer-risk-score-likelihood.json"
        )

    def predict(self):
        """
        Your algorithm goes here
        """

        # read image
        # image = sitk.ReadImage(str(self.image_input_path))
        clinical_info = self.clinical_info
        print("Clinical info: ")
        print(clinical_info)

        # TODO: Add your inference code here

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)
        transform = transforms.Compose(
            [
                transforms.LoadImage(image_only=True),
                transforms.EnsureChannelFirst(channel_dim=None),
                transforms.Resize(
                    spatial_size=(128, 128, 32),
                    mode=("trilinear"),
                ),
                transforms.ScaleIntensity(minv=0.0, maxv=1.0),
                transforms.NormalizeIntensity(),
            ]
        )
        image = transform(str(self.image_input_path))
        image = image.unsqueeze(0)
        model = SwinViTMultimodal_v3(
            n_classes=2,
            img_size=(128, 128, 32),
            in_channels=1,
            in_num_features=2,
        )
        model.load_state_dict(torch.load("model_best_state.pth", map_location=device))
        model.to(device)
        image = image.to(device)
        clinical_info = torch.tensor(
            [clinical_info["patient_age"], clinical_info["psa"]], dtype=torch.float32
        ).to(device)
        risk_score_likelihood = softmax(model(image, clinical_info.view(1, -1)), dim=1)[
            0
        ][1].item()
        # our code generates a random probability
        print(image.shape)
        # risk_score_likelihood = random.random()
        if risk_score_likelihood > 0.5:
            risk_score = "High"
        else:
            risk_score = "Low"
        print("Risk score: ", risk_score)
        print("Risk score likelihood: ", risk_score_likelihood)

        # save case-level class
        with open(str(self.risk_score_output_file), "w") as f:
            json.dump(risk_score, f)

        # save case-level likelihood
        with open(str(self.risk_score_likelihood_output_file), "w") as f:
            json.dump(float(risk_score_likelihood), f)


if __name__ == "__main__":
    print(torch.__version__)
    Prostatecancerriskprediction().predict()
