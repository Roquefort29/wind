from django.contrib import admin
from .models import Country, Year, TotalCumulativeInstalledCapacity, GrowthRate

class CountryAdmin(admin.ModelAdmin):
    list_display = ('name', 'land_area')
    search_fields = ('name',)
    list_filter = ('name',)
    fields = ('name', 'flag_photo', 'country_description', 'land_area')

class YearAdmin(admin.ModelAdmin):
    list_display = ('year',)
    search_fields = ('year',)
    list_filter = ('year',)

class TotalCumulativeInstalledCapacityAdmin(admin.ModelAdmin):
    list_display = ('country', 'year', 'value')
    search_fields = ('country__name', 'year__year')
    list_filter = ('country', 'year')

class GrowthRateAdmin(admin.ModelAdmin):
    list_display = ('country', 'year', 'value')
    search_fields = ('country__name', 'year__year')
    list_filter = ('country', 'year')

admin.site.register(Country, CountryAdmin)
admin.site.register(Year, YearAdmin)
admin.site.register(TotalCumulativeInstalledCapacity, TotalCumulativeInstalledCapacityAdmin)
admin.site.register(GrowthRate, GrowthRateAdmin)
