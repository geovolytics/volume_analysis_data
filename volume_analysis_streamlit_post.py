import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial import Delaunay
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import Image

# Title
st.set_page_config(page_title="Volume Computation", page_icon="📊💧")
st.title("Volume Calculation and Visualization Tool")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
Welcome to the **Volume Analysis and Visualization Tool**. 
This application showcases 3D surface visualizations and volume calculations using preloaded datasets.
Explore the data, visualizations, and results effortlessly.
""")

# Load the CSV files directly from GitHub
pre_surface_data = pd.read_csv("PRE_DREDGE_csv.csv")
post_surface_data = pd.read_csv("POST_DREDGE_csv.csv")

# Display the data in Streamlit
if pre_surface_data is not None and post_surface_data is not None:
    st.write("")
    st.subheader("Preview of Uploaded Data")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Reference Surface Data ({len(pre_surface_data)} points)")
        st.write(pre_surface_data.head())
    with col2:
        st.text(f"Post Surface Data ({len(post_surface_data)} points)")
        st.write(post_surface_data.head())

    x = pre_surface_data['eastings'].values
    y = pre_surface_data['northings'].values
    z = pre_surface_data['height'].values

    xx = post_surface_data['eastings'].values
    yy = post_surface_data['northings'].values
    zz = post_surface_data['height'].values

    x_ref, y_ref, z_ref = pre_surface_data['eastings'].values, pre_surface_data['northings'].values, pre_surface_data['height'].values
    x_vol, y_vol, z_vol = post_surface_data['eastings'].values, post_surface_data['northings'].values, post_surface_data['height'].values

    x_pre, y_pre, z_pre = pre_surface_data['eastings'].values, pre_surface_data['northings'].values, pre_surface_data['height'].values
    x_post, y_post, z_post = post_surface_data['eastings'].values, post_surface_data['northings'].values, post_surface_data['height'].values

    st.write("")
    st.write("")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pre Surface Map", "Post Surface Map", "Both Surfaces Map", "Volume Analysis", "Comparison with Civil 3D Result"])

    # Display Pre Surface
    with tab1:
        points = np.vstack((x, y)).T
        tri = Delaunay(points)

        fig = go.Figure()

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            color='blue',
            opacity=0.7
        ))

        fig.update_layout(
            title="Zoomable 3D Reference Surface Map (Pre)",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                aspectratio=dict(x=1.5, y=2, z=1),
                aspectmode='manual',
                camera=dict(
                    eye=dict(x=3, y=3, z=2.5)
                ),
            ),
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
        pio.write_html(fig, file="3d_surface_map_pre.html", auto_open=False)

    # Display Post Surface
    with tab2:
        points = np.vstack((xx, yy)).T
        tri = Delaunay(points)

        fig = go.Figure()

        fig.add_trace(go.Mesh3d(
            x=xx,
            y=yy,
            z=zz,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            color='red',
            opacity=0.7
        ))

        fig.update_layout(
            title="Zoomable 3D Post Surface Map",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                aspectratio=dict(x=1.5, y=2, z=1),
                aspectmode='manual',
                camera=dict(
                    eye=dict(x=3, y=3, z=2.5)
                ),
            ),
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
        pio.write_html(fig, file="3d_surface_map_post.html", auto_open=False)

    with tab3:
        points_ref = np.vstack((x_ref, y_ref)).T
        tri_ref = Delaunay(points_ref)

        points_vol = np.vstack((x_vol, y_vol)).T
        tri_vol = Delaunay(points_vol)

        fig = go.Figure()

        # Add reference surface
        fig.add_trace(go.Mesh3d(
            x=x_ref,
            y=y_ref,
            z=z_ref,
            i=tri_ref.simplices[:, 0],
            j=tri_ref.simplices[:, 1],
            k=tri_ref.simplices[:, 2],
            color='blue',
            opacity=0.8,
            name='Reference Surface'
        ))

        # Add volume surface
        fig.add_trace(go.Mesh3d(
            x=x_vol,
            y=y_vol,
            z=z_vol,
            i=tri_vol.simplices[:, 0],
            j=tri_vol.simplices[:, 1],
            k=tri_vol.simplices[:, 2],
            color='red',
            opacity=0.8,
            name='Volume Surface'
        ))

        fig.update_layout(
            title="Zoomable Reference and Volume Surfaces (TIN)",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectratio=dict(x=1.5, y=2, z=1),
                aspectmode='manual',
                camera=dict(
                    eye=dict(x=3, y=3, z=2.5)
                ),
            ),
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
        pio.write_html(fig, file="TIN_surface_maps.html", auto_open=False)


    def extend_reference_surface(pre_x, pre_y, pre_z, post_x, post_y, method='linear'):
        """
        Extends the pre-surface to cover the entire post-surface area by extrapolating
        the edge values of the pre-surface.

        Parameters:
        pre_x, pre_y, pre_z: coordinates of the pre-surface
        post_x, post_y: coordinates of the post-surface extent
        method: extrapolation method ('linear', 'nearest', or 'edge')

        Returns:
        Extended pre-surface Z values for the given post-surface points
        """
        if method == 'edge':
            from scipy.spatial import cKDTree
            pre_points = np.column_stack((pre_x, pre_y))
            kdtree = cKDTree(pre_points)

            distances, indices = kdtree.query(np.column_stack((post_x, post_y)))
            extended_z = pre_z[indices]
        else:
            extended_z = griddata(
                points=(pre_x, pre_y),
                values=pre_z,
                xi=(post_x, post_y),
                method=method,
                fill_value=np.nan
            )

            if np.any(np.isnan(extended_z)):
                mask = np.isnan(extended_z)
                extended_z[mask] = griddata(
                    points=(pre_x, pre_y),
                    values=pre_z,
                    xi=(post_x[mask], post_y[mask]),
                    method='nearest'
                )

        return extended_z


    def check_boundary_overlap(pre_coords, post_coords):
        """
        Check if post-surface extends beyond pre-surface boundaries

        Returns:
        - needs_extension: boolean indicating if extension is needed
        - overlap_ratio: percentage of post points within pre boundary
        """
        pre_x, pre_y, _ = pre_coords
        post_x, post_y, _ = post_coords

        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        pre_points = np.column_stack((pre_x, pre_y))
        pre_hull = ConvexHull(pre_points)
        pre_hull_path = Path(pre_points[pre_hull.vertices])

        post_points = np.column_stack((post_x, post_y))
        points_inside = pre_hull_path.contains_points(post_points)

        overlap_ratio = np.sum(points_inside) / len(post_points) * 100
        needs_extension = not np.all(points_inside)

        return needs_extension, overlap_ratio


    def calculate_volume_with_boundary_check(pre_coords, post_coords, grid_size=1.0):
        """
        Calculate volume between surfaces with intelligent boundary handling
        """
        pre_x, pre_y, pre_z = pre_coords
        post_x, post_y, post_z = post_coords

        # Check if extension is needed
        needs_extension, overlap_ratio = check_boundary_overlap(pre_coords, post_coords)

        # Create regular grid
        x_min = min(np.min(pre_x), np.min(post_x))
        x_max = max(np.max(pre_x), np.max(post_x))
        y_min = min(np.min(pre_y), np.min(post_y))
        y_max = max(np.max(pre_y), np.max(post_y))

        x_grid = np.arange(x_min, x_max + grid_size, grid_size)
        y_grid = np.arange(y_min, y_max + grid_size, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate post-surface
        post_points = np.column_stack((post_x, post_y))
        Z_post = griddata(post_points, post_z, (X, Y), method='linear')

        # Create mask for post-surface valid area
        hull_post = ConvexHull(post_points)
        post_path = Path(post_points[hull_post.vertices])
        post_mask = post_path.contains_points(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)

        # Initialize pre-surface grid
        Z_pre = np.full_like(Z_post, np.nan)
        grid_points = np.column_stack((X[post_mask].ravel(), Y[post_mask].ravel()))

        if needs_extension:
            notice = f"Notice: Post-surface extends beyond pre-surface boundaries."
            ratio = f"Overlap ratio: {overlap_ratio:.1f}% of post points within pre-surface"
            # Use extension method for points outside pre-surface
            Z_pre[post_mask] = extend_reference_surface(
                pre_x, pre_y, pre_z,
                grid_points[:, 0], grid_points[:, 1],
                method='linear'
            )
        else:
            notice = "All post-surface points within pre-surface boundaries."
            ratio = "Using direct interpolation without extension."
            # Direct interpolation is sufficient
            Z_pre[post_mask] = griddata(
                np.column_stack((pre_x, pre_y)),
                pre_z,
                grid_points,
                method='linear'
            )

        # Calculate volume differences
        Z_diff = Z_post - Z_pre
        cell_area = grid_size * grid_size

        cut_mask = (Z_diff < 0) & post_mask
        fill_mask = (Z_diff > 0) & post_mask

        cut_volume = abs(np.sum(Z_diff[cut_mask])) * cell_area
        fill_volume = np.sum(Z_diff[fill_mask]) * cell_area
        net_volume = fill_volume - cut_volume
        total_area = np.sum(post_mask) * cell_area

        return {
            'cut_volume': cut_volume,
            'fill_volume': fill_volume,
            'net_volume': net_volume,
            'total_area': total_area,
            'grid_data': (X, Y, Z_diff, post_mask),
            'cut_mask': cut_mask,
            'fill_mask': fill_mask,
            'needs_extension': needs_extension,
            'overlap_ratio': overlap_ratio,
            'notice': notice,
            'ratio': ratio
        }


    # Volume Calculation
    with tab4:
        # Perform volume calculation
        results = calculate_volume_with_boundary_check(
            (x_pre, y_pre, z_pre),
            (x_post, y_post, z_post),
            grid_size=1.0  # Adjust grid size if needed
        )

        # Display results to the user in a formatted way
        st.subheader("Volume Calculation Results")
        st.write(f"{results['notice']}")
        st.write(f"{results['ratio']}")
        st.write(f"**Cut Volume:** {results['cut_volume']:,.3f} cubic meters")
        st.write(f"**Fill Volume:** {results['fill_volume']:,.3f} cubic meters")
        st.write(f"**Net Volume Change:** {results['net_volume']:,.3f} cubic meters")
        st.write(f"**Total Affected Area:** {results['total_area']:,.3f} square meters")

    # Hardcoded Civil 3D results (replace these with actual values)
    civil3d_cut_volume = 306.96
    civil3d_fill_volume = 33748.54
    civil3d_Net_volume = 33441.58

    # Calculate differences
    cut_difference = abs(results['cut_volume'] - civil3d_cut_volume)
    fill_difference = abs(results['fill_volume'] - civil3d_fill_volume)
    net_volume_difference = abs(results['net_volume'] - civil3d_Net_volume)

    with tab5:
        # Display the Civil 3D image
        image = Image.open("C3D Volume.PNG")  # Replace with your image file path
        st.image(image, caption="Cut and Fill Results from Civil 3D", use_column_width=True)

        # Display Civil 3D results and differences
        st.subheader("Results Comparison")
        st.markdown(f"""
            - **Cut Volume (Civil 3D):** {civil3d_cut_volume:,.3f} cubic meters  
            - **Fill Volume (Civil 3D):** {civil3d_fill_volume:,.3f} cubic meters  
            - **Net_volume (Civil 3D):** {civil3d_Net_volume:,.3f} cubic meters  
            - **Cut Volume Difference:** {cut_difference:,.3f} cubic meters  
            - **Fill Volume Difference:** {fill_difference:,.3f} cubic meters  
            - **Net Volume Difference:** {net_volume_difference:,.3f} cubic meters 
            """)


else:  # closing the if statement
    st.info("CSV file refuse to load.")
