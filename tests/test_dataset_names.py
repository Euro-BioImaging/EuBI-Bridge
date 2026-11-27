"""Outputs of image sets and plates named after the dataset (Workplan E29): Matthew's
Viventis batches each hold an OME companion named ome-tiff.companion.ome, so every output
was ome-tiff.ome.zarr and they collided. A set opened through an index file takes the name
its metadata gives (a one-image set's image name, a one-plate file's plate name), unless the
name is generic or another input of the same output folder shares it."""
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eubi_bridge.utils.path_utils import (dataset_names, disambiguate_output_names,
                                          usable_dataset_name)

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _viventis(folder: Path, image_name: str, Z=2, value=1, named=True):
    """A Viventis-like batch: ``ome-tiff.companion.ome`` naming one image (no ``Name``
    when not `named`), and one plane file per z (each pointing back at the companion)."""
    folder.mkdir(parents=True, exist_ok=True)
    this = f"urn:uuid:companion-{image_name}"
    tiffdata = []
    for z in range(Z):
        name = f"t0001_488_plane{z + 1:04d}.tif"
        uuid = f"urn:uuid:{image_name}-{z}"
        stub = (f'<OME xmlns="{NS}" UUID="{uuid}"><BinaryOnly '
                f'MetadataFile="ome-tiff.companion.ome" UUID="{this}"/></OME>')
        tifffile.imwrite(folder / name, np.full((16, 20), value + z, "u2"), description=stub,
                         metadata=None)
        tiffdata.append(f'<TiffData FirstT="0" FirstC="0" FirstZ="{z}" IFD="0" PlaneCount="1">'
                        f'<UUID FileName="{name}">{uuid}</UUID></TiffData>')
    name_attr = f' Name="{image_name}"' if named else ""
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}" UUID="{this}">'
           f'<Image ID="Image:0"{name_attr}><Pixels ID="Pixels:0" '
           f'DimensionOrder="XYZCT" Type="uint16" SizeX="20" SizeY="16" SizeZ="{Z}" SizeC="1" '
           f'SizeT="1" PhysicalSizeX="0.5" PhysicalSizeY="0.5"><Channel ID="Channel:0:0" '
           f'SamplesPerPixel="1"/>{"".join(tiffdata)}</Pixels></Image></OME>')
    (folder / "ome-tiff.companion.ome").write_text(xml, encoding="utf-8")
    return folder / "ome-tiff.companion.ome"


@pytest.mark.parametrize("name, kept", [
    ("Position 11_Settings 1", "Position 11_Settings 1"),
    ("0459 1A", "0459 1A"),
    ("a/b:c", "a_b_c"),
    ("Plate01", None), ("Microplate", None), ("Image:0", None), ("Series 0", None),
    ("Sequence:1", None),                      # a Bruker companion's image (E8)
    ("Position 1", None), ("", None), ("cf91602d-0ed3-4cc7-b5c2-6374492106ab", None),
])
def test_generic_names_name_nothing(name, kept):
    assert usable_dataset_name(name) == kept


def test_a_companion_set_takes_its_image_name(tmp_path):
    path = _viventis(tmp_path / "Position 11_Settings 1", "Position 11_Settings 1")
    assert dataset_names([str(path)]) == {str(path): "Position 11_Settings 1"}


@pytest.mark.parametrize("image_name, named", [("Image:0", True), ("unused", False)])
def test_a_companion_without_an_image_name_keeps_its_file_name(tmp_path, image_name, named):
    """Workplan E8: an OME companion whose image has a generic name, or none, is named
    after the companion file, without its suffix: 'ome-tiff', never
    'ome-tiff.companion' (micro-reader's name for an image of no name of its own, which
    wrote 'ome-tiff.companion.ome.zarr')."""
    path = str(_viventis(tmp_path / "Embryo_A_run3", image_name, named=named))
    assert dataset_names([path]) == {}
    assert disambiguate_output_names([path]) == {path: "ome-tiff"}


def test_a_companion_converts_under_its_file_name(tmp_path):
    """End to end (E8): an unnamed companion image converts once, as 'ome-tiff.ome.zarr',
    with the planes the companion lists."""
    from eubi_bridge.ebridge import EuBIBridge
    import zarr
    path = _viventis(tmp_path / "in" / "Embryo_A_run3", "unused", Z=3, value=5, named=False)
    EuBIBridge().to_zarr(str(path), str(tmp_path / "out"), use_threading=True, verbose=False)
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["ome-tiff.ome.zarr"]
    level = np.asarray(zarr.open_group(str(tmp_path / "out" / "ome-tiff.ome.zarr"),
                                       mode="r")["0"])
    assert level.squeeze().tolist() == [np.full((16, 20), 5 + z).tolist() for z in range(3)]


def test_viventis_batches_no_longer_collide(tmp_path):
    """Three batches converted in one run: one output each, named after its position."""
    from eubi_bridge.ebridge import EuBIBridge
    for k, pos in enumerate(("Position 1_Settings 1", "Position 2_Settings 1",
                             "Position 11_Settings 1")):
        _viventis(tmp_path / "in" / pos, pos, value=10 * k)
    EuBIBridge().to_zarr(str(tmp_path / "in"), str(tmp_path / "out"), use_threading=True,
                         verbose=False)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "Position 11_Settings 1.ome.zarr", "Position 1_Settings 1.ome.zarr",
        "Position 2_Settings 1.ome.zarr"]


def test_a_shared_name_keeps_the_names_inputs_had(tmp_path):
    """Two sets naming the same dataset (two measurements of one plate): neither takes
    the shared name; each keeps its own, told apart as before."""
    a = _viventis(tmp_path / "run1" / "Position 7_Settings 1", "Position 7_Settings 1")
    b = _viventis(tmp_path / "run2" / "Position 7_Settings 1", "Position 7_Settings 1")
    from eubi_bridge.ebridge import _output_places
    import pandas as pd
    df = pd.DataFrame({"input_path": [str(a), str(b)], "output_path": [str(tmp_path / "o")] * 2})
    _, names = _output_places(df, str(tmp_path / "o"), [], False)
    # as before E29: the folders that tell them apart, then the file's own name
    assert sorted(names.values()) == ["run1_Position 7_Settings 1_ome-tiff",
                                      "run2_Position 7_Settings 1_ome-tiff"]


def test_single_image_files_keep_their_file_names(tmp_path):
    tifffile.imwrite(tmp_path / "cells.ome.tif", np.zeros((8, 8), "u2"), ome=True,
                     metadata={"Name": "something else"})
    assert dataset_names([str(tmp_path / "cells.ome.tif")]) == {}
    assert disambiguate_output_names([str(tmp_path / "cells.ome.tif")]) == {
        str(tmp_path / "cells.ome.tif"): "cells"}
