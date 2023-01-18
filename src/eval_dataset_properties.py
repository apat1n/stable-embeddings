from src.dataset import Dataset
from postprocess_data import MinInteractionsFilter, IdsEncoder

if __name__ == '__main__':
    data = Dataset('bcr').get_data()

    min_iterations_filter = MinInteractionsFilter()
    ids_encoder = IdsEncoder()

    data = min_iterations_filter.transform(data)
    data = ids_encoder.fit_transform(data)

    n_users = len(data['user_id'].unique())
    n_items = len(data['item_id'].unique())
    n_interactions = len(data)
    print(f'users: {n_users}')
    print(f'items: {n_items}')
    print(f'interactions: {n_interactions}')
    print(f'density: {(n_interactions / n_users / n_items) * 100:.2f}%')

    print(data)
