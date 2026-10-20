# Changelog

## 0.1.7 (unreleased)

Pin release: requires micro-reader>=0.0.3,<0.1 (micro-reader 0.1.0 renames the image index 'scene' to 'series', which this release does not read), ome_zarr_pyramid[remote]>=0.1.0,<0.2 and dyna-zarr>=0.1.0,<0.2. No other change. The first release that reads the new index is eubi-bridge / eubi-bridge-lite 0.3.0.

ozp is bounded below 0.2, not 0.3: the 0.1 line was built and tested against ome_zarr_pyramid 0.1.0. With ozp 0.2.x, the ozp writer places the eubi_bridge provenance block one level deeper in the root zarr.json than eubi-bridge's own writer (tests/test_ozp_writer.py::test_ozp_v3_sharded_matches fails with 0.2.0 and 0.2.1, passes with 0.1.0).
