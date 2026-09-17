"""Primary downloadable documents at links actually published by their owners."""
import fetch_sources as fetch
fetch.OUT=fetch.ROOT/"output/breakthrough_20260916/research_gap/documents/sources"
fetch.SOURCES={
 "mit_queue.pdf":"https://web.mit.edu/hamsa/www/pubs/SimaiakisBalakrishnan_TS2014.pdf",
 "mit_tandem.pdf":"https://web.mit.edu/hamsa/www/pubs/BadrinathBalakrishnanACC2017.pdf",
 "mit_kalman.pdf":"https://web.mit.edu/hamsa/www/pubs/KhadilkarBalakrishnanATIO2011a.pdf",
 "mit_queue_trajectory.pdf":"https://web.mit.edu/hamsa/www/pubs/LeeSimaiakisBalakrishnanDASC2010.pdf",
 "eurocontrol_metadata.pdf":"https://www.eurocontrol.int/sites/default/files/2025-04/eurocontrol-aviation-data-repository-research-metadata.pdf",
 "eurocontrol_dpi.pdf":"https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf",
 "graph_attention.pdf":"https://mdpi-res.com/d_attachment/aerospace/aerospace-11-00371/article_deploy/aerospace-11-00371.pdf",
 "graph_roadlevel.pdf":"https://mdpi-res.com/d_attachment/aerospace/aerospace-12-00721/article_deploy/aerospace-12-00721.pdf",
}
if __name__=="__main__":fetch.main()
