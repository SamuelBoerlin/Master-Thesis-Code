import objaverse
import pickle
import os.path
import signal

if __name__ == '__main__':
	def load_downloaded_uids() -> set[str]:
		if os.path.isfile("downloaded_uids.pkl"):
			with open('downloaded_uids.pkl', 'rb') as f:
				return pickle.load(f)
		return set()

	def save_downloaded_uids(uids: set[str]):
		with open('downloaded_uids.pkl', 'wb') as f:
			pickle.dump(uids, f)

	downloaded_uids = load_downloaded_uids()

	uids = [id for id in objaverse.load_uids() if id not in downloaded_uids]

	def download_batch(uids: list[str], processes: int):
		objects = objaverse.load_objects(
			uids=uids,
			download_processes=processes
		)
		downloaded_uids.update(uids)
		save_downloaded_uids(downloaded_uids)
		return objects

	def download_objaverse():
		processes = 32
		batch_size = 1000
		start = 0
		while start < len(uids) - 1:
			end = min(start + batch_size, len(uids) - 1)
			print("Download slice: " + str(start) + ":" + str(end))
			print(download_batch(uids[start:end], processes))
			start = end

	download_objaverse()

