"""
Interactive Data Visualizer
----------------------------
A Streamlit app for uploading, cleaning, filtering, and visualizing
tabular data with interactive Plotly charts.

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Interactive Data Visualizer",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    "<h1 style='margin-bottom:0;'>📊 Interactive Data Visualizer</h1>"
    "<p style='color:gray;margin-top:0;'>Upload, clean, filter, and explore your data — all in one place.</p>",
    unsafe_allow_html=True,
)

@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


@st.cache_data
def load_excel(file_or_path):
    return pd.read_excel(file_or_path)


def get_summary(data: pd.DataFrame) -> dict:
    return {
        "rows": data.shape[0],
        "cols": data.shape[1],
        "missing": int(data.isnull().sum().sum()),
        "duplicates": int(data.duplicated().sum()),
    }


def show_summary_metrics(data: pd.DataFrame, label: str):
    s = get_summary(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", s["rows"])
    c2.metric("Columns", s["cols"])
    c3.metric("Missing Values", s["missing"])
    c4.metric("Duplicate Rows", s["duplicates"])
    st.caption(label)


BINARY_TRUE_VALUES = {"yes", "y", "true", "t", "1", "1.0"}
BINARY_FALSE_VALUES = {"no", "n", "false", "f", "0", "0.0"}


def detect_binary_columns(data: pd.DataFrame) -> list:
    binary_cols = []
    for col in data.columns:
        values = data[col].dropna().astype(str).str.strip().str.lower().unique()
        values_set = set(values)
        if 0 < len(values_set) <= 2 and values_set.issubset(BINARY_TRUE_VALUES | BINARY_FALSE_VALUES):
            binary_cols.append(col)
    return binary_cols


def add_encoded_binary_columns(data: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
 
    data = data.copy()
    for col in binary_cols:
        encoded_name = f"{col} (encoded)"
        data[encoded_name] = data[col].astype(str).str.strip().str.lower().map(
            lambda v: 1 if v in BINARY_TRUE_VALUES else (0 if v in BINARY_FALSE_VALUES else np.nan)
        )
    return data

NONE_OPTION = "— None —"  # avoids clashing with a real column literally named "None"

st.sidebar.header("📁 Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = load_csv(uploaded_file)
        else:
            df = load_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Couldn't read this file: {e}")

if df is not None:

    # Fingerprint the file so a new upload resets "original_df" correctly
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("original_file_id") != file_id:
        st.session_state.original_df = df.copy()
        st.session_state.original_file_id = file_id
        st.session_state.pop("filled_df", None)

    df = df.copy()

    tab_overview, tab_clean, tab_visualize, tab_insights = st.tabs(
        ["🔍 Overview", "🧹 Clean & Filter", "📈 Visualize", "🧠 Insights"]
    )

    with tab_overview:
        show_summary_metrics(df, "Raw dataset summary")

        with st.expander("Preview data", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("Column details"):
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Missing": df.isnull().sum().values,
                "Unique Values": df.nunique().values,
            })
            st.caption("Click a row to see exactly where that column's missing values are.")
            selection = st.dataframe(
                info_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="column_details_table",
            )

            selected_rows = selection.selection.rows if selection is not None else []
            if selected_rows:
                selected_col = info_df.iloc[selected_rows[0]]["Column"]
                missing_count = int(info_df.iloc[selected_rows[0]]["Missing"])
                if missing_count > 0:
                    st.markdown(f"**Rows where `{selected_col}` is missing** ({missing_count} total):")
                    missing_rows = df[df[selected_col].isnull()]
                    st.dataframe(missing_rows, use_container_width=True)
                else:
                    st.success(f"`{selected_col}` has no missing values.")

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        detected_binary = detect_binary_columns(df)
        if detected_binary:
            st.info(
                "Binary-style columns detected (Yes/No, Y/N, True/False, or 0/1): "
                + ", ".join(f"`{c}`" for c in detected_binary)
                + ". Head to the **Clean & Filter** tab to standardize them for numeric analysis."
            )

    
    with tab_clean:
        st.subheader("Data Cleaning")

        if st.button("♻️ Reset to original data"):
            df = st.session_state.original_df.copy()
            st.session_state.pop("filled_df", None)
            st.success("Data reset to original state.")

        rows_with_missing = df[df.isnull().any(axis=1)]
        if not rows_with_missing.empty:
            with st.expander(f"🔎 Rows with missing values ({len(rows_with_missing)} rows)", expanded=True):
                cols_with_missing = df.columns[df.isnull().any()].tolist()
                only_affected_cols = st.checkbox("Only show columns that have missing values", value=True)
                display_df = rows_with_missing[cols_with_missing] if only_affected_cols else rows_with_missing
                st.dataframe(display_df, use_container_width=True)
        else:
            st.success("No missing values in this dataset.")

        col1, col2 = st.columns(2)

        with col1:
            missing_choice = st.radio(
                "Handle missing values",
                ["Leave as is", "Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"],
            )
            if missing_choice == "Drop rows":
                df = df.dropna()
            elif missing_choice == "Fill with Mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_choice == "Fill with Median":
                df = df.fillna(df.median(numeric_only=True))
            elif missing_choice == "Fill with Mode":
                df = df.fillna(df.mode().iloc[0])
            elif missing_choice == "Fill with Custom Value":
                cols_with_missing_now = df.columns[df.isnull().any()].tolist()
                if not cols_with_missing_now:
                    st.info("No missing values left to fill.")
                else:
                    st.caption("Set a fill value for each column — different columns usually need different values.")
                    target_cols = st.multiselect(
                        "Columns to fill",
                        cols_with_missing_now,
                        default=cols_with_missing_now,
                    )
                    fill_values = {}
                    for c in target_cols:
                        col_default = "0" if pd.api.types.is_numeric_dtype(df[c]) else ""
                        fill_values[c] = st.text_input(
                            f"Value for `{c}`", value=col_default, key=f"fill_custom_{c}"
                        )

                    if st.button("✅ Apply "):
                        for c, custom_value in fill_values.items():
                            if custom_value != "":
                                try:
                                    df[c] = df[c].fillna(float(custom_value))
                                except ValueError:
                                    df[c] = df[c].fillna(custom_value)
                        st.session_state.filled_df = df.copy()
                        st.success(f"Filled missing values in: {', '.join(target_cols)}")

                    if "filled_df" in st.session_state:
                        df = st.session_state.filled_df.copy()

        with col2:
            if st.checkbox("Remove duplicate rows"):
                df = df.drop_duplicates()

        st.divider()
        st.subheader("Filter Data")

        filter_col = st.selectbox("Filter by column (optional)", [NONE_OPTION] + df.columns.tolist())

        if filter_col != NONE_OPTION:
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                col_data = df[filter_col].dropna()
                if col_data.empty:
                    st.info(f"`{filter_col}` has no non-missing values to filter on.")
                elif col_data.min() == col_data.max():
                    st.info(f"`{filter_col}` has a single constant value ({col_data.min()}); nothing to filter.")
                else:
                    min_v, max_v = float(col_data.min()), float(col_data.max())
                    selected_range = st.slider("Range", min_v, max_v, (min_v, max_v))
                    df = df[df[filter_col].between(*selected_range)]
            else:
                options = df[filter_col].dropna().unique().tolist()
                selected_values = st.multiselect("Values to keep", options, default=options)
                df = df[df[filter_col].isin(selected_values)]

        st.divider()
        st.subheader("Standardize Yes/No-style Columns")

        binary_cols = detect_binary_columns(df)

        if binary_cols:
            st.caption(
                "These columns look binary (Yes/No, Y/N, True/False, or 0/1). "
                "Standardizing adds a numeric 0/1 version so they work in correlation, "
                "aggregation, and numeric charts — no matter how they were originally entered."
            )
            st.write(", ".join(f"`{c}`" for c in binary_cols))
            standardize = st.checkbox("Standardize these columns", value=True)
            if standardize:
                df = add_encoded_binary_columns(df, binary_cols)
        else:
            st.caption("No binary-style (Yes/No, 0/1, True/False) columns detected.")

        show_summary_metrics(df, "Dataset after cleaning & filtering")
        st.dataframe(df.head(20), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download cleaned CSV", csv_bytes, "cleaned_data.csv", "text/csv")

    with tab_visualize:
        st.subheader("Build a Chart")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        chart_type = st.selectbox(
            "Chart type",
            [
                "Line", "Bar", "Scatter", "Histogram",
                "Box", "Violin", "Pie", "Area", "Scatter Matrix",
            ],
        )

        colA, colB, colC = st.columns(3)
        with colA:
            x_axis = st.selectbox("X-axis", all_cols)
        with colB:
            y_axis = st.selectbox("Y-axis", [NONE_OPTION] + all_cols)
        with colC:
            color_by = st.selectbox("Color by (optional)", [NONE_OPTION] + all_cols)

        color_arg = None if color_by == NONE_OPTION else color_by
        theme = st.radio("Theme", ["plotly", "plotly_dark", "seaborn"], horizontal=True)

        fig = None
        try:
            if chart_type == "Line":
                fig = px.line(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color=color_arg)
            elif chart_type == "Box":
                fig = px.box(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg)
            elif chart_type == "Violin":
                fig = px.violin(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg, box=True)
            elif chart_type == "Pie":
                fig = px.pie(df, names=x_axis, values=None if y_axis == NONE_OPTION else y_axis)
            elif chart_type == "Area":
                fig = px.area(df, x=x_axis, y=None if y_axis == NONE_OPTION else y_axis, color=color_arg)
            elif chart_type == "Scatter Matrix":
                if len(numeric_cols) >= 2:
                    fig = px.scatter_matrix(df, dimensions=numeric_cols[:5], color=color_arg)
                else:
                    st.warning("Need at least 2 numeric columns for a scatter matrix.")

            if fig is not None:
                fig.update_layout(template=theme, title=f"{chart_type} Chart")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Couldn't build that chart with the selected columns: {e}")

        st.divider()
        st.subheader("Group & Aggregate")

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            group_col = st.selectbox("Group by", [NONE_OPTION] + all_cols, key="group_col")
        with gcol2:
            agg_col = st.selectbox("Aggregate column", [NONE_OPTION] + numeric_cols, key="agg_col")
        with gcol3:
            agg_func = st.selectbox("Function", ["mean", "sum", "count", "min", "max"], key="agg_func")

        if group_col != NONE_OPTION and agg_col != NONE_OPTION:
            grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
            st.dataframe(grouped, use_container_width=True)
            bar_fig = px.bar(grouped, x=group_col, y=agg_col, title=f"{agg_func.title()} of {agg_col} by {group_col}")
            st.plotly_chart(bar_fig, use_container_width=True)

    with tab_insights:
        st.subheader("Correlation Heatmap")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            heat_fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(heat_fig, use_container_width=True)

            st.subheader("Strongest Correlations")
            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            corr_pairs.columns = ["Column A", "Column B", "Correlation"]
            corr_pairs = corr_pairs.reindex(
                corr_pairs["Correlation"].abs().sort_values(ascending=False).index
            ).head(10)
            st.dataframe(corr_pairs, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns to compute correlations.")

        st.divider()
        st.subheader("Quick Auto-Insights")

        insights = []
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append(f"**{col}** is skewed to the {direction} (skewness = {skew:.2f}).")
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        for col, pct in missing_pct.items():
            if pct > 20:
                insights.append(f"**{col}** has {pct:.1f}% missing values — consider addressing this.")

        if insights:
            for i in insights:
                st.markdown(f"- {i}")
        else:
            st.info("No major issues detected in the current dataset.")

else:
    st.info("📂 Upload a CSV or Excel file from the sidebar to get started.")