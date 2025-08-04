import pandas as pd

def create_probability_matrix(prediction_file, level='class'):
    """
    Create a project-technology matrix with probabilities.
    Duplicates are grouped and their probabilities are stored as lists.

    Args:
        prediction_file: Path to CSV file with predictions
        level: 'section' or 'class' (default: 'class')

    Returns:
        DataFrame: Matrix with unique projects as rows and technology codes as columns
                  Values are lists of probabilities (for handling duplicates)
    """
    df = pd.read_csv(prediction_file)

    project_predictions = {}

    for _, row in df.iterrows():
        project_id = row['project_id']

        if project_id not in project_predictions:
            project_predictions[project_id] = {}

        if level == 'section':
            predicted_col = 'section_predicted'
            probability_col = 'section_probability'
        else:
            predicted_col = 'class_predicted'
            probability_col = 'class_probability'


        if pd.notna(row.get(predicted_col, '')) and row[predicted_col]:
            codes = [c.strip() for c in str(row[predicted_col]).split(',')]
            probs = [float(p.strip()) for p in str(row[probability_col]).split(',')]


            for code, prob in zip(codes, probs):
                if code not in project_predictions[project_id]:
                    project_predictions[project_id][code] = []
                project_predictions[project_id][code].append(prob)

    all_codes = set()
    for predictions in project_predictions.values():
        all_codes.update(predictions.keys())

    all_codes = sorted(all_codes, key=lambda x: (len(x), x))

    matrix_data = []
    for project_id, predictions in project_predictions.items():
        row = {'project_id': project_id}
        for code in all_codes:
            if code in predictions:
                row[code] = predictions[code]
            else:
                row[code] = []
        matrix_data.append(row)

    matrix = pd.DataFrame(matrix_data).set_index('project_id')

    return matrix


def get_unique_labels(matrix):

    summary = []
    for project_id in matrix.index:
        # Get all non-empty predictions
        predicted_codes = []
        for code in matrix.columns:
            if len(matrix.loc[project_id, code]) > 0:
                predicted_codes.append(code)

        summary.append({
            'project_id': project_id,
            'unique_labels': ', '.join(predicted_codes),
            'num_labels': len(predicted_codes),
            'num_predictions': len(matrix.loc[project_id, matrix.columns[0]]) if len(predicted_codes) > 0 else 0
        })

    return pd.DataFrame(summary)


def label_distribution(matrix):
    label_counts = {}
    for col in matrix.columns:

        count = sum(1 for val in matrix[col] if val != '' and val != [])
        label_counts[col] = count


    project_label_counts = {}
    for idx in matrix.index:
        count = sum(1 for val in matrix.loc[idx] if val != '' and val != [])
        project_label_counts[idx] = count

    print("=== LABEL DISTRIBUTION ===\n")

    print("1. Labels by frequency:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(matrix)) * 100
        print(f"   {label}: {count} projects ({pct:.1f}%)")


    print("\n2. Number of labels per project:")
    for project, count in project_label_counts.items():
        print(f"   {project}: {count} labels")


    label_counts_list = list(project_label_counts.values())
    print("\n3. Summary:")
    print(f"    Average labels per project: {sum(label_counts_list) / len(label_counts_list):.2f}")
    print(f"   Max labels in a project: {max(label_counts_list)}")
    print(f"   Min labels in a project: {min(label_counts_list)}")
    print(f"   Projects with no labels: {sum(1 for c in label_counts_list if c == 0)}")

    print("\n4. Common label combinations:")
    combinations = {}
    for idx in matrix.index:
        labels = []
        for col in matrix.columns:
            if matrix.loc[idx, col] != '' and matrix.loc[idx, col] != []:
                labels.append(col)
        if len(labels) > 1:
            combo = ', '.join(sorted(labels))
            combinations[combo] = combinations.get(combo, 0) + 1

    for combo, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
        print(f"{combo}: {count} projects")


def summary(matrix):
    for col in matrix.columns:
        count = sum(1 for val in matrix[col] if val != '' and val != [])
        print(f"{col}: {count}/{len(matrix)} projects ({count / len(matrix) * 100:.0f}%)")


if __name__ == "__main__":
    class_matrix = create_probability_matrix("data/new_predictions/model_predictions_threshold.csv", level='class')
    print("Class-level Matrix:")
    print(class_matrix)

    class_summary = get_unique_labels(class_matrix)
    print("Unique labels per project:")
    print(class_summary)

    section_matrix = create_probability_matrix("data/new_predictions/model_predictions_threshold.csv", level='section')
    print(section_matrix)

    class_matrix.to_csv("project_technology_matrix_class.csv")
    section_matrix.to_csv("project_technology_matrix_section.csv")
    label_distribution(section_matrix)

    print("\n=== SUMMARY ===")
    summary(section_matrix)