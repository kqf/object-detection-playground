import pytest

from detection.dataset import DetectionDataset, read_data
from detection.augmentations import transform
from detection.model import build_model

from detection.plot import plot


@pytest.fixture
def max_epochs(request):
    return request.config.getoption("--max-epochs")


def test_model(fake_dataset, max_epochs, fixed_seed):
    df = read_data(fake_dataset / "train.csv")
    print(df.head())
    train = DetectionDataset(df, fake_dataset, transforms=transform())

    model = build_model(max_epochs=max_epochs, top_n=5)
    model.fit(train)
    model.save_params(f_params='test-params.pkl')

    preds = model.predict(train)
    first_image_pred = preds[0][:, :4]
    first_image, targets, _ = train.example(0)
    print("Target coordinates:")
    print(targets)

    # first_image = torch.ones((3, 460, 460))
    # for first_image in train:
    #     first_image_pred = torch.zeros(10, 4)
    #     first_image = first_image[0]

    #     # first_image = train[0][0] * 0 + 1
    #     first_image_pred = torch.ones((10, 4))
    #     first_image_pred[:, 0] = 0.4
    #     first_image_pred[:, 1] = 0.4
    #     first_image_pred[:, 2] = 0.2
    #     first_image_pred[:, 3] = 0.2

    print("preds")
    print(first_image_pred)

    plot((first_image, first_image_pred, targets),
         convert_bbox=True, ofile='dummy-test.png')
