"""Two inputs sharing a filename must not target the same output store.

Outputs are named after the input's basename, so ``A/img.tif`` and ``B/img.tif``
both resolve to ``img.zarr``.  With ``overwrite=False`` the second job fails;
with ``overwrite=True`` it silently destroys the first.  Colliding names take a
parent-directory prefix instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.utils.path_utils import disambiguate_output_names, output_stem
from eubi_bridge.conversion.conversion_worker import _generate_output_path


class TestDisambiguation:
    def test_distinct_names_are_untouched(self):
        """A batch with no collision must keep the paths it already had."""
        paths = ["/a/one.tif", "/b/two.tif"]
        assert disambiguate_output_names(paths) == {
            "/a/one.tif": "one", "/b/two.tif": "two"}

    def test_collision_takes_parent_prefix(self):
        result = disambiguate_output_names(["/a/img.tif", "/b/img.tif"])
        assert result == {"/a/img.tif": "a_img", "/b/img.tif": "b_img"}

    def test_only_colliding_names_change(self):
        result = disambiguate_output_names(
            ["/a/img.tif", "/b/img.tif", "/c/unique.tif"])
        assert result["/c/unique.tif"] == "unique"
        assert result["/a/img.tif"] == "a_img"

    def test_shared_parent_walks_further_up(self):
        """A common parent cannot disambiguate; keep climbing until it does."""
        result = disambiguate_output_names(["/r1/s/img.tif", "/r2/s/img.tif"])
        assert result == {"/r1/s/img.tif": "r1_img", "/r2/s/img.tif": "r2_img"}

    def test_three_way_collision(self):
        result = disambiguate_output_names(
            ["/a/img.tif", "/b/img.tif", "/c/img.tif"])
        assert sorted(result.values()) == ["a_img", "b_img", "c_img"]

    def test_multi_dot_extension_stem(self):
        """Matches _generate_output_path: a two-part suffix is removed whole."""
        result = disambiguate_output_names(
            ["/a/image.ome.tiff", "/b/image.ome.tiff"])
        assert sorted(result.values()) == ["a_image", "b_image"]

    def test_dots_in_the_name_tell_files_apart(self):
        """Until 0.3.2 both were 'Image_V4' (everything before the first dot) and the
        second took its suffix: 'Image_V4_1_b_vsi'."""
        a, b = r"C:\s\Image_V4.1_BF.vsi", r"C:\s\Image_V4.1_FL_3D.vsi"
        assert disambiguate_output_names([a, b]) == {a: "Image_V4.1_BF", b: "Image_V4.1_FL_3D"}

    def test_results_are_unique(self):
        paths = ["/a/x/img.tif", "/b/x/img.tif", "/c/y/img.tif", "/d/other.tif"]
        values = list(disambiguate_output_names(paths).values())
        assert len(set(values)) == len(values)

    def test_duplicate_path_terminates(self):
        """The same file listed twice cannot be told apart — must not hang."""
        result = disambiguate_output_names(["/a/img.tif", "/a/img.tif"])
        assert result == {"/a/img.tif": "img"}

    def test_windows_separators(self):
        result = disambiguate_output_names(
            [r"C:\data\A\img.tif", r"C:\data\B\img.tif"])
        assert sorted(result.values()) == ["A_img", "B_img"]

    def test_windows_separators_on_a_posix_host(self, monkeypatch):
        """A batch table written on Windows may be run on Linux.

        ``os.path`` only treats ``\`` as a separator on Windows, so without
        explicit handling the whole path became the name on Linux.  Simulating
        posixpath catches that from any host.
        """
        import posixpath
        from eubi_bridge.utils import path_utils
        monkeypatch.setattr(path_utils.os, "path", posixpath)
        result = disambiguate_output_names(
            [r"C:\data\A\img.tif", r"C:\data\B\img.tif"])
        assert sorted(result.values()) == ["A_img", "B_img"]

    def test_posix_separators_on_a_posix_host(self, monkeypatch):
        import posixpath
        from eubi_bridge.utils import path_utils
        monkeypatch.setattr(path_utils.os, "path", posixpath)
        result = disambiguate_output_names(
            ["/data/A/img.tif", "/data/B/img.tif"])
        assert sorted(result.values()) == ["A_img", "B_img"]

    def test_mixed_separators(self):
        """CSV rows can mix separators after path joining."""
        result = disambiguate_output_names(
            [r"C:\data\A/img.tif", "/data/B/img.tif"])
        assert sorted(result.values()) == ["A_img", "B_img"]

    def test_real_world_case(self):
        """The reported failure: two Study folders, one image name."""
        a = "/demo/S-BIAD1047/Images/Study_22/image_344_Mitochondria.ome.tiff"
        b = "/demo/other/Study_99/image_344_Mitochondria.ome.tiff"
        result = disambiguate_output_names([a, b])
        assert result[a] == "Study_22_image_344_Mitochondria"
        assert result[b] == "Study_99_image_344_Mitochondria"
        assert result[a] != result[b]


class TestSameFolder:
    """Files of one folder sharing a name but not a suffix: their parents cannot tell
    them apart, so the suffix does (an Airyscan .czi and its .ims). Before, the
    parents were walked up to the drive: 'C:_name', no valid folder name."""

    def test_suffix_tells_them_apart(self):
        a, b = r"C:\data\run\x_Airyscan.czi", r"C:\data\run\x_Airyscan.ims"
        assert disambiguate_output_names([a, b]) == {a: "x_Airyscan_czi",
                                                     b: "x_Airyscan_ims"}

    def test_the_other_files_keep_their_names(self):
        paths = [r"C:\d\r\a.czi", r"C:\d\r\a.ims", r"C:\d\r\a_proc.czi", r"C:\d\r\b.czi"]
        result = disambiguate_output_names(paths)
        assert result == {paths[0]: "a_czi", paths[1]: "a_ims", paths[2]: "a_proc",
                          paths[3]: "b"}

    def test_a_drive_is_never_a_prefix(self):
        a, b = r"C:\img.tif", r"D:\img.tif"
        result = disambiguate_output_names([a, b])
        assert all(":" not in name for name in result.values())
        assert len(set(result.values())) == 2


class TestPlateIndexNames:
    """A plate index file with a fixed name (Harmony Index.idx.xml / index.xml,
    CellVoyager MeasurementData.mlf) is named after its measurement folder, above
    Images/; two such plates in one batch stay apart by that folder's parents."""

    def test_harmony_index_takes_the_measurement_folder(self):
        path = "/data/Plate 1-Measurement1/Images/Index.idx.xml"
        assert disambiguate_output_names([path]) == {path: "Plate 1-Measurement1"}

    def test_hs_archive_index_takes_the_measurement_folder(self):
        path = "/data/hs/3a1d3fc3/images/index.xml"
        assert disambiguate_output_names([path]) == {path: "3a1d3fc3"}

    def test_cellvoyager_index_takes_its_folder(self):
        path = "/data/run7/MeasurementData.mlf"
        assert disambiguate_output_names([path]) == {path: "run7"}

    def test_two_plates_of_one_name_take_the_parent_above(self):
        a, b = "/x/M1/Images/Index.idx.xml", "/y/M1/Images/Index.idx.xml"
        assert disambiguate_output_names([a, b]) == {a: "x_M1", b: "y_M1"}

    def test_an_image_called_index_keeps_its_name(self):
        assert disambiguate_output_names(["/a/index.tif"]) == {"/a/index.tif": "index"}

    @pytest.mark.parametrize("path, name", [
        ("/data/plate_001/experiment_descriptor.xml", "plate_001"),               # scanR
        ("/data/export/tif/ImageIndex.ColumbusIDX.xml", "tif"),                    # Columbus
        ("/data/MFG_2205/MFG_2205_A01f00d0.C01", "MFG_2205"),                      # Cellomics
        ("/data/run_171942/WellB05_ChannelDIA_Seq0000.nd2", "run_171942"),         # NIS run
        ("/data/run/WellA01_PointA01_0003_ChannelFITC BP_Seq0003.nd2", "run"),
    ])
    def test_a_plate_whose_file_names_nothing_takes_its_folder(self, path, name):
        assert disambiguate_output_names([path]) == {path: name}

    def test_plate_files_written_per_image_keep_their_names(self):
        """as_plate off: each file of a run is an output of its own."""
        files = ["/data/run/WellB05_ChannelDIA_Seq0000.nd2", "/data/run/WellB06_ChannelDIA_Seq0001.nd2"]
        assert disambiguate_output_names(files, per_image=files) == {
            files[0]: "WellB05_ChannelDIA_Seq0000", files[1]: "WellB06_ChannelDIA_Seq0001"}

    def test_other_nd2_files_keep_their_names(self):
        assert disambiguate_output_names(["/data/x/Well plate 3.nd2"]) == {
            "/data/x/Well plate 3.nd2": "Well plate 3"}


class TestOutputStem:
    @pytest.mark.parametrize("path, stem", [
        ("/a/img.tif", "img"),
        ("/a/img.ome.tiff", "img"),
        ("/a/IMG.OME.TIF", "IMG"),
        (r"C:\a\Image_V4.1_BF.vsi", "Image_V4.1_BF"),
        ("/a/run 2.5x.czi", "run 2.5x"),
        ("/a/two  spaces.ome.tif", "two  spaces"),
        ("/a/plate.ome.zarr", "plate"),
        ("/a/plate.ome.zarr/", "plate"),
        ("/a/IN_02.r3d_d3d", "IN_02"),
        ("/a/no_suffix", "no_suffix"),
        ("/a/.ome.tif", ".ome"),
    ])
    def test_the_suffix_only_is_removed(self, path, stem):
        assert output_stem(path) == stem


class TestGeneratedPath:
    def test_default_naming_unchanged(self):
        assert _generate_output_path("/out", "/a/img.ome.tiff") == "/out/img.zarr"

    def test_a_dot_in_the_name_stays(self):
        assert _generate_output_path("/out", "/a/Image_V4.1_BF.vsi", 2) == \
            "/out/Image_V4.1_BF_scene2.zarr"

    def test_resolved_basename_replaces_derived_name(self):
        assert _generate_output_path(
            "/out", "/a/img.ome.tiff", resolved_basename="A_img") == "/out/A_img.zarr"

    @pytest.mark.parametrize("scene,tile,expected", [
        (2, None, "/out/A_img_scene2.zarr"),
        (None, 5, "/out/A_img_tile5.zarr"),
        (2, 5, "/out/A_img_scene2_tile5.zarr"),
    ])
    def test_suffixes_compose_with_resolved_basename(self, scene, tile, expected):
        """Disambiguation must not disturb multi-scene/tile naming."""
        assert _generate_output_path(
            "/out", "/a/img.tif", scene, tile, resolved_basename="A_img") == expected
