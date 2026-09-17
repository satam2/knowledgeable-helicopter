"""Independently verify the expanded historical airport geometry cache."""
import verify_geometry as verify

verify.PUBLIC=verify.external_path(verify.ROOT/"output/breakthrough_20260916/geometry/public_v2")
verify.OUT=verify.external_path(verify.ROOT/"private_runs/breakthrough_20260916/geometry_v2")

if __name__=="__main__":verify.main()
