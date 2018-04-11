from abc import ABCMeta, abstractmethod

class SubnetBase:
	__metaclass__ = ABCMeta
	@abstractmethod
	def Build(self):
		pass
