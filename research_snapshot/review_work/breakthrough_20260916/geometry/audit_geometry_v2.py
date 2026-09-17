"""Run the preserved geometry audit against expanded historical bounds."""
import json
import audit_geometry as audit

audit.PUBLIC=audit.external_path(audit.ROOT/"output/breakthrough_20260916/geometry/public_v2")
audit.OUT=audit.external_path(audit.ROOT/"private_runs/breakthrough_20260916/geometry_v2")

if __name__=="__main__":
    receipt=json.loads((audit.PUBLIC/"receipt.json").read_text())
    assert receipt["status"]=="complete"
    for airport,record in receipt["airports"].items():assert audit.sha(audit.PUBLIC/(airport+".json"))==record["sha256"]
    audit.main()
    (audit.OUT/"invocation.json").write_text(json.dumps({"wrapper_sha256":audit.sha(__file__),"core_source_sha256":audit.sha(audit.__file__),"public_receipt_sha256":audit.sha(audit.PUBLIC/"receipt.json"),"public_path":str(audit.PUBLIC),"output_path":str(audit.OUT)},indent=2),encoding="utf-8")
