import io
import unittest
import zipfile
from inspect_workbooks import rows,shared_strings,excel_date,sheets


class WorkbookParserTests(unittest.TestCase):
    def archive(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,"w") as z:
            z.writestr("xl/sharedStrings.xml",'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>APT_ICAO</t></si><si><r><t>LI</t></r><r><t>RF</t></r></si></sst>')
            z.writestr("xl/worksheets/sheet1.xml",'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="D1" t="inlineStr"><is><t>FLIGHT_DATE</t></is></c></row><row r="7"><c r="A7" t="s"><v>1</v></c><c r="C7"><v>0</v></c><c r="D7"><v>46234</v></c><c r="F7"><f>C7*2</f><v>0</v></c><c r="H7"/></row></sheetData></worksheet>')
            z.writestr("xl/workbook.xml",'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr/><sheets><sheet name="DATA" r:id="rId1"/></sheets></workbook>')
            z.writestr("xl/_rels/workbook.xml.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_sparse_cells_strings_cached_formula_zero_and_blank(self):
        with self.archive() as z:
            strings=shared_strings(z)
            self.assertEqual(strings,["APT_ICAO","LIRF"])
            path=sheets(z)["DATA"]
            records=list(rows(z,path,strings))
        self.assertEqual(records[1][0],7)
        self.assertEqual(records[1][1],{"A7":"LIRF","C7":0.,"D7":46234.,"F7":0.,"H7":None})
        self.assertEqual(records[1][2],{"F7":"C7*2"})
        self.assertNotIn("B7",records[1][1])

    def test_declared_modern_excel_dates(self):
        self.assertEqual(excel_date(43466),"2019-01-01")
        self.assertEqual(excel_date(46234),"2026-07-31")
        self.assertEqual(excel_date(46248),"2026-08-14")
        self.assertIsNone(excel_date(None))


if __name__=="__main__":unittest.main()
