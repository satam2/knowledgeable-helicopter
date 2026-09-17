"""Follow verified primary links; preserve discovery failures separately."""
import fetch_sources as fetch
fetch.OUT=fetch.ROOT/"output/breakthrough_20260916/research_gap/primary/sources"
fetch.SOURCES={
 "mit_publications":"https://web.mit.edu/hamsa/www/publications.html",
 "eurocontrol_structure":"https://www.eurocontrol.int/publication/aviation-data-research-structure-and-sample",
 "eurocontrol_dpi":"https://www.eurocontrol.int/publication/departure-planning-information-dpi-implementation-guide",
 "eurocontrol_airport_data":"https://ansperformance.eu/methodology/additional-taxi-out-time/",
 "eurocontrol_airport_data_intro":"https://ansperformance.eu/data/",
 "graph_attention":"https://www.mdpi.com/2226-4310/11/5/371",
 "graph_roadlevel":"https://www.mdpi.com/2226-4310/12/8/721",
 "graph_aiaa":"https://arc.aiaa.org/doi/10.2514/1.D0369",
 "queue_informs":"https://pubsonline.informs.org/doi/10.1287/trsc.2015.0603",
}
if __name__=="__main__":fetch.main()
