import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("tracts_with_residuals.geojson")

# Use the actual residual column name:
RES_COL = "residuals"

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
gdf.plot(
    column=RES_COL,
    legend=True,
    edgecolor="black",
    linewidth=0.1,
    ax=ax
)

ax.set_title("Poisson GLM Deviance Residuals")
ax.axis("off")
plt.show()